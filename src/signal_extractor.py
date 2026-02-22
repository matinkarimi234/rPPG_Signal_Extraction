import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from scipy.signal import butter, filtfilt, welch, detrend, windows

# =============================
# Config
# =============================
@dataclass
class RPPGConfig:
    f_lo: float = 0.7          # Hz
    f_hi: float = 4.0          # Hz
    bp_order: int = 3
    pos_win_sec: float = 1.6
    pos_overlap: float = 0.5   # hop = win*(1-overlap)
    hr_win_sec: float = 10.0
    hr_hop_sec: float = 1.0
    min_valid_ratio: float = 0.7  # minimum fraction of finite samples required
    snr_guard_hz: float = 0.15    # "peak neighborhood" for SNR numerator
    snr_min_db: float = 0.0       # reject window if SNR below this

CFG = RPPGConfig()

# =============================
# Utilities
# =============================
def interp_nans_1d(x: np.ndarray, min_valid_ratio=0.7) -> np.ndarray:
    """Linear-interpolate NaNs in 1D; if too many NaNs -> return all-NaN."""
    x = np.asarray(x, float)
    n = len(x)
    good = np.isfinite(x)
    if good.mean() < min_valid_ratio:
        return np.full_like(x, np.nan)
    if good.all():
        return x
    t = np.arange(n)
    y = x.copy()
    y[~good] = np.interp(t[~good], t[good], x[good])
    return y

def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    mu = np.nanmean(x)
    sd = np.nanstd(x) + 1e-12
    return (x - mu) / sd

def bandpass(x: np.ndarray, fs: float, f_lo: float, f_hi: float, order: int) -> np.ndarray:
    """Zero-phase Butterworth bandpass."""
    x = np.asarray(x, float)
    nyq = 0.5 * fs
    b, a = butter(order, [f_lo/nyq, f_hi/nyq], btype="band")
    return filtfilt(b, a, x)

def safe_detrend(x):
    x = np.asarray(x, float)
    if not np.all(np.isfinite(x)):
        return x
    return detrend(x, type="linear")

# =============================
# rPPG methods
# =============================
def rppg_green(rgb: np.ndarray, fs: float, cfg: RPPGConfig = CFG) -> np.ndarray:
    """Baseline: green channel bandpassed."""
    g = interp_nans_1d(rgb[:, 1], cfg.min_valid_ratio)
    if not np.all(np.isfinite(g)):
        return np.full(len(g), np.nan)
    g = safe_detrend(g)
    return bandpass(g, fs, cfg.f_lo, cfg.f_hi, cfg.bp_order)

def rppg_chrom(rgb: np.ndarray, fs: float, cfg: RPPGConfig = CFG) -> np.ndarray:
    """
    CHROM (De Haan & Jeanne).
    Normalize channels, form X/Y, then combine.
    """
    R = interp_nans_1d(rgb[:, 0], cfg.min_valid_ratio)
    G = interp_nans_1d(rgb[:, 1], cfg.min_valid_ratio)
    B = interp_nans_1d(rgb[:, 2], cfg.min_valid_ratio)
    if not (np.all(np.isfinite(R)) and np.all(np.isfinite(G)) and np.all(np.isfinite(B))):
        return np.full(len(R), np.nan)

    Rn = R / (np.mean(R) + 1e-12)
    Gn = G / (np.mean(G) + 1e-12)
    Bn = B / (np.mean(B) + 1e-12)

    X = 3*Rn - 2*Gn
    Y = 1.5*Rn + Gn - 1.5*Bn

    X = bandpass(safe_detrend(X), fs, cfg.f_lo, cfg.f_hi, cfg.bp_order)
    Y = bandpass(safe_detrend(Y), fs, cfg.f_lo, cfg.f_hi, cfg.bp_order)

    alpha = np.std(X) / (np.std(Y) + 1e-12)
    S = X - alpha * Y
    return zscore(S)

def rppg_pos(rgb: np.ndarray, fs: float, cfg: RPPGConfig = CFG) -> np.ndarray:
    """
    POS (Wang et al.).
    Efficient overlap-add (hop > 1) instead of sliding by 1 sample.
    """
    R = interp_nans_1d(rgb[:, 0], cfg.min_valid_ratio)
    G = interp_nans_1d(rgb[:, 1], cfg.min_valid_ratio)
    B = interp_nans_1d(rgb[:, 2], cfg.min_valid_ratio)
    if not (np.all(np.isfinite(R)) and np.all(np.isfinite(G)) and np.all(np.isfinite(B))):
        return np.full(len(R), np.nan)

    C = np.vstack([R, G, B]).T.astype(float)
    C = C / (np.mean(C, axis=0, keepdims=True) + 1e-12)

    n = len(C)
    win = max(5, int(round(cfg.pos_win_sec * fs)))
    hop = max(1, int(round(win * (1.0 - cfg.pos_overlap))))
    hann = windows.hann(win, sym=False)

    out = np.zeros(n, float)
    wsum = np.zeros(n, float)

    P = np.array([[0,  1, -1],
                  [-2, 1,  1]], dtype=float)  # 2x3

    for start in range(0, n - win + 1, hop):
        seg = C[start:start+win]
        seg = seg - np.mean(seg, axis=0, keepdims=True)
        X = seg @ P.T  # (win,2)

        x1 = X[:, 0]
        x2 = X[:, 1]
        alpha = np.std(x1) / (np.std(x2) + 1e-12)
        s = x1 - alpha * x2
        s = s - np.mean(s)

        out[start:start+win] += s * hann
        wsum[start:start+win] += hann

    out = out / np.maximum(wsum, 1e-12)
    out = bandpass(safe_detrend(out), fs, cfg.f_lo, cfg.f_hi, cfg.bp_order)
    return zscore(out)

# =============================
# HR + Quality metrics
# =============================
def welch_hr_and_snr(x: np.ndarray, fs: float, cfg: RPPGConfig = CFG):
    """
    Returns bpm, snr_db, freqs, psd.
    SNR is computed as:
      10*log10( power around peak / power in band excluding that neighborhood )
    """
    x = np.asarray(x, float)
    if not np.all(np.isfinite(x)):
        return np.nan, -np.inf, None, None

    f, p = welch(x, fs=fs, nperseg=min(len(x), int(8*fs)))
    band = (f >= cfg.f_lo) & (f <= cfg.f_hi)
    if band.sum() < 3:
        return np.nan, -np.inf, f, p

    fb = f[band]
    pb = p[band]
    i0 = int(np.argmax(pb))
    f0 = float(fb[i0])
    bpm = 60.0 * f0

    # SNR in-band
    guard = cfg.snr_guard_hz
    num = (fb >= f0-guard) & (fb <= f0+guard)
    den = ~num
    p_num = float(np.sum(pb[num]) + 1e-12)
    p_den = float(np.sum(pb[den]) + 1e-12)
    snr_db = 10.0 * np.log10(p_num / p_den)

    return bpm, snr_db, f, p

def hr_track_sliding(x: np.ndarray, fs: float, cfg: RPPGConfig = CFG):
    win = int(round(cfg.hr_win_sec * fs))
    hop = int(round(cfg.hr_hop_sec * fs))
    times, bpms, snrs = [], [], []
    for start in range(0, len(x) - win + 1, hop):
        seg = x[start:start+win]
        bpm, snr_db, _, _ = welch_hr_and_snr(seg, fs, cfg)
        times.append((start + win/2) / fs)
        bpms.append(bpm if snr_db >= cfg.snr_min_db else np.nan)
        snrs.append(snr_db)
    return np.array(times), np.array(bpms), np.array(snrs)

# =============================
# IO + Analysis
# =============================
def load_npz(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    fs = float(d["fps"]) if "fps" in d else 30.0
    ts = d["timestamps"].astype(float) if "timestamps" in d else np.arange(len(d["forehead"])) / fs

    rois = {}
    for k in ["forehead", "left_cheek", "right_cheek"]:
        if k in d:
            rois[k] = d[k].astype(float)  # (T,3) RGB
    # optional quality extras from improved extractor
    roi_pixels = d["roi_pixels"] if "roi_pixels" in d else None
    face_ok = d["face_ok"] if "face_ok" in d else None
    return fs, ts, rois, roi_pixels, face_ok

def analyze_one(npz_path: str, method="POS", cfg: RPPGConfig = CFG, show=True, save_dir=None):
    fs, t, rois, roi_pixels, face_ok = load_npz(npz_path)
    method = method.upper()
    rppg_fn = {"GREEN": rppg_green, "CHROM": rppg_chrom, "POS": rppg_pos}[method]

    results = {}
    for roi_name, rgb in rois.items():
        s = rppg_fn(rgb, fs, cfg)

        bpm, snr_db, f, pxx = welch_hr_and_snr(s, fs, cfg)
        tt, bpm_track, snr_track = hr_track_sliding(s, fs, cfg)

        results[roi_name] = dict(
            t=t, rgb=rgb, rppg=s,
            bpm=bpm, snr_db=snr_db,
            f=f, pxx=pxx,
            hr_t=tt, hr_bpm=bpm_track, hr_snr_db=snr_track
        )

        if show or save_dir:
            title = f"{Path(npz_path).name} | ROI={roi_name} | {method} | HR≈{bpm:.1f} bpm | SNR≈{snr_db:.1f} dB"

            # 1) RGB
            plt.figure(figsize=(10, 3))
            plt.plot(t, interp_nans_1d(rgb[:,0], cfg.min_valid_ratio), label="R")
            plt.plot(t, interp_nans_1d(rgb[:,1], cfg.min_valid_ratio), label="G")
            plt.plot(t, interp_nans_1d(rgb[:,2], cfg.min_valid_ratio), label="B")
            plt.title("Raw ROI mean RGB — " + roi_name)
            plt.xlabel("Time (s)")
            plt.ylabel("Mean intensity (a.u.)")
            plt.legend(ncol=3, fontsize=8)
            plt.tight_layout()
            if save_dir:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                plt.savefig(Path(save_dir)/f"rgb_{roi_name}.png", dpi=200)
            if show: plt.show()
            else: plt.close()

            # 2) rPPG waveform
            plt.figure(figsize=(10, 3))
            plt.plot(t, s)
            plt.title("rPPG (bandpassed) — " + title)
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude (z)")
            plt.tight_layout()
            if save_dir:
                plt.savefig(Path(save_dir)/f"rppg_{roi_name}.png", dpi=200)
            if show: plt.show()
            else: plt.close()

            # 3) PSD
            if f is not None:
                plt.figure(figsize=(10, 3))
                plt.semilogy(f, pxx)
                plt.xlim(0, 6)
                plt.title("Welch PSD — " + title)
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("PSD")
                plt.tight_layout()
                if save_dir:
                    plt.savefig(Path(save_dir)/f"psd_{roi_name}.png", dpi=200)
                if show: plt.show()
                else: plt.close()

            # 4) HR track + SNR track (separate figures)
            plt.figure(figsize=(10, 3))
            plt.plot(tt, bpm_track)
            plt.ylim(40, 200)
            plt.title("HR track (Welch windows) — " + roi_name)
            plt.xlabel("Time (s)")
            plt.ylabel("HR (bpm)")
            plt.tight_layout()
            if save_dir:
                plt.savefig(Path(save_dir)/f"hr_{roi_name}.png", dpi=200)
            if show: plt.show()
            else: plt.close()

            plt.figure(figsize=(10, 3))
            plt.plot(tt, snr_track)
            plt.title("Quality track: SNR(dB) — " + roi_name)
            plt.xlabel("Time (s)")
            plt.ylabel("SNR (dB)")
            plt.tight_layout()
            if save_dir:
                plt.savefig(Path(save_dir)/f"snr_{roi_name}.png", dpi=200)
            if show: plt.show()
            else: plt.close()

    # Best ROI selection (use SNR, then bpm validity)
    best = None
    best_score = -np.inf
    for roi_name, d in results.items():
        score = d["snr_db"]
        if np.isfinite(d["bpm"]) and score > best_score:
            best_score = score
            best = roi_name

    if best is not None:
        print(f"Best ROI by SNR: {best} | HR≈{results[best]['bpm']:.1f} bpm | SNR≈{results[best]['snr_db']:.1f} dB | fs={fs:.2f}")

    return results

def analyze_folder(folder="data/processed", method="POS", cfg: RPPGConfig = CFG):
    folder = Path(folder)
    files = sorted(folder.glob("*.npz"))
    if not files:
        print("No .npz files found in", folder)
        return

    print(f"Found {len(files)} files in {folder}")
    for f in files:
        res = analyze_one(str(f), method=method, cfg=cfg, show=False, save_dir=None)
        # Best already printed

# ---- Example usage ----
if __name__ == "__main__":
    analyze_one("data/processed/1_ppg_data.npz", method="GREEN", show=True, save_dir=None)
    # analyze_folder("data/processed", method="POS")
