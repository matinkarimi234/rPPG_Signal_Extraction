import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from scipy.signal import butter, filtfilt, welch, detrend

# -----------------------------
# Utilities
# -----------------------------
def interp_nans(x: np.ndarray) -> np.ndarray:
    """Linear-interpolate NaNs in a 1D array."""
    x = x.astype(float)
    n = len(x)
    t = np.arange(n)
    good = np.isfinite(x)
    if good.sum() < max(5, n // 10):
        return np.full_like(x, np.nan)
    x2 = x.copy()
    x2[~good] = np.interp(t[~good], t[good], x[good])
    return x2

def bandpass(x, fs, f_lo=0.7, f_hi=4.0, order=3):
    """Zero-phase Butterworth bandpass."""
    nyq = 0.5 * fs
    b, a = butter(order, [f_lo/nyq, f_hi/nyq], btype="band")
    return filtfilt(b, a, x)

def zscore(x):
    x = np.asarray(x, float)
    mu = np.nanmean(x)
    sd = np.nanstd(x) + 1e-12
    return (x - mu) / sd

# -----------------------------
# rPPG methods
# -----------------------------
def rppg_green(rgb, fs):
    """Simplest baseline: use G channel only."""
    g = interp_nans(rgb[:, 1])
    g = detrend(g, type='linear')
    return bandpass(g, fs)

def rppg_chrom(rgb, fs):
    """
    CHROM method (De Haan & Jeanne).
    Works best with decent lighting, stable ROI.
    """
    R = interp_nans(rgb[:, 0])
    G = interp_nans(rgb[:, 1])
    B = interp_nans(rgb[:, 2])

    # Normalize (remove intensity scale)
    Rn = R / (np.mean(R) + 1e-12)
    Gn = G / (np.mean(G) + 1e-12)
    Bn = B / (np.mean(B) + 1e-12)

    X = 3*Rn - 2*Gn
    Y = 1.5*Rn + Gn - 1.5*Bn

    X = detrend(X, type='linear')
    Y = detrend(Y, type='linear')

    Xf = bandpass(X, fs)
    Yf = bandpass(Y, fs)

    alpha = np.std(Xf) / (np.std(Yf) + 1e-12)
    S = Xf - alpha * Yf
    return zscore(S)

def rppg_pos(rgb, fs, win_sec=1.6):
    """
    POS method (Wang et al.) – sliding window projection.
    Often very robust.
    """
    R = interp_nans(rgb[:, 0])
    G = interp_nans(rgb[:, 1])
    B = interp_nans(rgb[:, 2])
    C = np.vstack([R, G, B]).T.astype(float)

    # Normalize per channel globally (simple, works ok)
    C = C / (np.mean(C, axis=0, keepdims=True) + 1e-12)

    n = len(C)
    win = int(round(win_sec * fs))
    win = max(win, 5)
    S = np.zeros(n, float)

    for t0 in range(0, n - win + 1):
        Cw = C[t0:t0+win]
        Cw = Cw - np.mean(Cw, axis=0, keepdims=True)

        # POS projection
        X = Cw @ np.array([[0, 1, -1],
                           [-2, 1, 1]], float).T   # shape (win,2)

        x1 = X[:, 0]
        x2 = X[:, 1]
        alpha = np.std(x1) / (np.std(x2) + 1e-12)
        s = x1 - alpha * x2

        S[t0:t0+win] += s

    S = detrend(S, type='linear')
    S = bandpass(S, fs)
    return zscore(S)

# -----------------------------
# HR estimation
# -----------------------------
def hr_fft(rppg, fs, f_lo=0.7, f_hi=4.0):
    """Single HR estimate from full segment using Welch peak."""
    rppg = np.asarray(rppg, float)
    if not np.all(np.isfinite(rppg)):
        return np.nan, (None, None)

    f, pxx = welch(rppg, fs=fs, nperseg=min(len(rppg), int(8*fs)))
    band = (f >= f_lo) & (f <= f_hi)
    if band.sum() < 3:
        return np.nan, (f, pxx)

    f_band = f[band]
    p_band = pxx[band]
    f0 = f_band[np.argmax(p_band)]
    bpm = 60.0 * f0
    return bpm, (f, pxx)

def hr_track_sliding(rppg, fs, win_sec=10.0, hop_sec=1.0, f_lo=0.7, f_hi=4.0):
    """Time-varying HR track with sliding Welch."""
    n = len(rppg)
    win = int(round(win_sec * fs))
    hop = int(round(hop_sec * fs))
    times = []
    bpms = []

    for start in range(0, n - win + 1, hop):
        seg = rppg[start:start+win]
        bpm, _ = hr_fft(seg, fs, f_lo, f_hi)
        times.append((start + win/2) / fs)
        bpms.append(bpm)

    return np.array(times), np.array(bpms)

# -----------------------------
# Main analysis
# -----------------------------
def load_npz(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    fps = float(d["fps"]) if "fps" in d else 30.0
    ts = d["timestamps"] if "timestamps" in d else np.arange(len(d["forehead"])) / fps

    rois = {}
    for k in ["forehead", "left_cheek", "right_cheek"]:
        if k in d:
            rois[k] = d[k].astype(float)  # shape (T,3) as (R,G,B)
    return fps, ts, rois

def analyze_one(npz_path: str, method="POS", show=True):
    fs, t, rois = load_npz(npz_path)
    out = {}

    method = method.upper()
    rppg_fn = {
        "GREEN": rppg_green,
        "CHROM": rppg_chrom,
        "POS": rppg_pos,
    }[method]

    for roi_name, rgb in rois.items():
        # Raw channels (handle NaNs)
        R = interp_nans(rgb[:,0]); G = interp_nans(rgb[:,1]); B = interp_nans(rgb[:,2])

        # rPPG
        s = rppg_fn(rgb, fs)

        # HR full + PSD
        bpm, (f, pxx) = hr_fft(s, fs)

        # HR track
        tt, bpm_track = hr_track_sliding(s, fs, win_sec=10.0, hop_sec=1.0)

        out[roi_name] = {
            "t": t, "rgb": np.vstack([R,G,B]).T, "rppg": s,
            "bpm": bpm, "f": f, "pxx": pxx,
            "hr_t": tt, "hr_bpm": bpm_track
        }

        if show:
            fig = plt.figure(figsize=(12, 8))
            fig.suptitle(f"{Path(npz_path).name} | ROI={roi_name} | method={method} | HR≈{bpm:.1f} bpm")

            ax1 = plt.subplot(3,1,1)
            ax1.plot(t, R, label="R")
            ax1.plot(t, G, label="G")
            ax1.plot(t, B, label="B")
            ax1.set_ylabel("Mean RGB")
            ax1.legend(loc="upper right")
            ax1.grid(True, alpha=0.3)

            ax2 = plt.subplot(3,1,2)
            ax2.plot(t, s)
            ax2.set_ylabel("rPPG (bandpassed)")
            ax2.set_xlabel("Time (s)")
            ax2.grid(True, alpha=0.3)

            ax3 = plt.subplot(3,2,5)
            ax3.plot(f, pxx)
            ax3.set_xlim(0, 6)
            ax3.set_xlabel("Frequency (Hz)")
            ax3.set_ylabel("PSD")
            ax3.grid(True, alpha=0.3)

            ax4 = plt.subplot(3,2,6)
            ax4.plot(tt, bpm_track)
            ax4.set_xlabel("Time (s)")
            ax4.set_ylabel("HR (bpm)")
            ax4.set_ylim(40, 200)
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

    return out

def analyze_folder(folder="data/processed", method="POS"):
    folder = Path(folder)
    files = sorted(folder.glob("*.npz"))
    if not files:
        print("No .npz files found in", folder)
        return

    summary = []
    for f in files:
        out = analyze_one(str(f), method=method, show=False)
        # pick best ROI = highest PSD peak in band (simple heuristic)
        best_roi = None
        best_score = -np.inf
        best_bpm = np.nan

        for roi_name, d in out.items():
            ff, pxx = d["f"], d["pxx"]
            if ff is None: 
                continue
            band = (ff >= 0.7) & (ff <= 4.0)
            score = np.max(pxx[band]) if band.any() else -np.inf
            if score > best_score:
                best_score = score
                best_roi = roi_name
                best_bpm = d["bpm"]

        summary.append((f.name, best_roi, best_bpm))

    print("\nSummary (best ROI per file):")
    for name, roi, bpm in summary:
        print(f"  {name:30s}  best={roi:12s}  HR≈{bpm:6.1f} bpm")

# ---- Example usage ----
if __name__ == "__main__":
    # analyze one file with plots:
    analyze_one("data/processed/508_ppg_data.npz", method="POS", show=True)

    # analyze all .npz in folder without plots:
    #analyze_folder("data/processed", method="POS")
