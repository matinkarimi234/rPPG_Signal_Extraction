"""
Option C extractor: multi-patch cheeks (2x2) + proportional forehead.

Outputs:
- forehead: (T,3) RGB means
- left_cheek_00..11, right_cheek_00..11: (T,3) RGB means
- left_cheek, right_cheek: aggregate (mean of patch means, NaN-safe)
- fps, frame_count, timestamps
- roi_pixels: (T,3) pixel counts for [forehead,left_cheek_agg,right_cheek_agg]
- roi_pixels_patches: (T,9) pixel counts for [forehead, L00,L01,L10,L11, R00,R01,R10,R11]
- face_ok: (T,) 0/1
- meta: dict
"""

import os
import cv2
import numpy as np
import urllib.request
from pathlib import Path
import argparse

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = "face_landmarker.task"


def ensure_model(path=MODEL_PATH, url=MODEL_URL):
    if not os.path.exists(path):
        print("Downloading face_landmarker.task...")
        urllib.request.urlretrieve(url, path)
        print("Download complete.")


# ---------------------------
# Landmark indices
# ---------------------------
BROW_INDICES = [46, 53, 52, 51, 48, 276, 283, 282, 281, 278]
LEFT_CHEEK_INDICES = [118, 119, 100, 136, 206, 205, 49, 203, 227, 137, 177, 215, 216]
RIGHT_CHEEK_INDICES = [346, 345, 374, 423, 424, 432, 279, 422, 426, 436, 434, 416, 376]

LEFT_TEMPLE = 127
RIGHT_TEMPLE = 356

NOSE_TIP = 1
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
CHIN = 152


# ---------------------------
# Skin mask (HSV or YCrCb)
# ---------------------------
def create_skin_mask_hsv(image_bgr, lower_hsv=(0, 20, 70), upper_hsv=(20, 255, 255)):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def create_skin_mask_ycrcb(image_bgr, cr=(133, 173), cb=(77, 127)):
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    _, Cr, Cb = cv2.split(ycrcb)
    mask = cv2.inRange(Cr, cr[0], cr[1]) & cv2.inRange(Cb, cb[0], cb[1])
    mask = mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


# ---------------------------
# ROI helpers
# ---------------------------
def clamp_rect(x1, y1, x2, y2, w, h):
    x1 = int(np.clip(x1, 0, w - 1))
    x2 = int(np.clip(x2, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    y2 = int(np.clip(y2, 0, h - 1))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def rect_mask(h, w, x1, y1, x2, y2):
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(m, (x1, y1), (x2, y2), 255, -1)
    return m


def shrink_rect(x, y, rw, rh, shrink=0.10):
    dx = int(round(rw * shrink))
    dy = int(round(rh * shrink))
    x2 = x + rw
    y2 = y + rh
    return x + dx, y + dy, x2 - dx, y2 - dy


def split_rect_2x2(x1, y1, x2, y2):
    """
    Returns dict: {"00": (x1,y1,xm,ym), "01": (xm,y1,x2,ym), "10": (x1,ym,xm,y2), "11": (xm,ym,x2,y2)}
    where 0=row top, 1=row bottom, 0=col left, 1=col right.
    """
    xm = (x1 + x2) // 2
    ym = (y1 + y2) // 2
    return {
        "00": (x1, y1, xm, ym),
        "01": (xm, y1, x2, ym),
        "10": (x1, ym, xm, y2),
        "11": (xm, ym, x2, y2),
    }


def landmarks_to_pts(landmarks, w, h):
    return np.array([(int(lm.x * w), int(lm.y * h)) for lm in landmarks], dtype=np.int32)


def safe_get_pt(pts, idx, fallback):
    if 0 <= idx < len(pts):
        return pts[idx]
    return np.array(fallback, dtype=np.int32)


def mean_rgb_in_mask(frame_bgr, mask):
    cnt = int(np.count_nonzero(mask))
    if cnt < 20:
        return (np.nan, np.nan, np.nan), cnt
    b, g, r, _ = cv2.mean(frame_bgr, mask=mask)
    return (float(r), float(g), float(b)), cnt


def nanmean_rgb(list_of_rgb):
    arr = np.array(list_of_rgb, dtype=float)  # (K,3)
    if np.all(~np.isfinite(arr)):
        return (np.nan, np.nan, np.nan)
    return tuple(np.nanmean(arr, axis=0).tolist())


def get_roi_masks_optionC(
    frame_bgr,
    pts,
    # Forehead sizing
    forehead_height_ratio=0.28,
    forehead_width_inset=0.08,
    # Cheek sizing / safety
    cheek_shrink=0.18,
    cheek_bottom_frac=0.45,
    nose_margin_px=12,
    mouth_margin_px=10,
    jaw_margin_px=18,
):
    """
    Returns masks for:
    - forehead
    - left_cheek_00..11
    - right_cheek_00..11
    And also aggregate cheek masks: left_cheek, right_cheek (union of patches)
    """
    h, w = frame_bgr.shape[:2]
    masks = {}

    # Reference points
    nose = safe_get_pt(pts, NOSE_TIP, (w // 2, h // 2))
    mouth_l = safe_get_pt(pts, MOUTH_LEFT, (w // 2 - 40, int(h * 0.65)))
    mouth_r = safe_get_pt(pts, MOUTH_RIGHT, (w // 2 + 40, int(h * 0.65)))
    chin = safe_get_pt(pts, CHIN, (w // 2, int(h * 0.85)))

    brow_y = int(np.min(pts[BROW_INDICES, 1])) if np.max(BROW_INDICES) < len(pts) else int(h * 0.35)
    chin_y = int(chin[1])
    face_h = max(20, chin_y - brow_y)

    # ---------- Forehead ----------
    left_t = safe_get_pt(pts, LEFT_TEMPLE, (int(w * 0.3), brow_y))
    right_t = safe_get_pt(pts, RIGHT_TEMPLE, (int(w * 0.7), brow_y))
    left_x = int(left_t[0])
    right_x = int(right_t[0])

    tw = max(10, right_x - left_x)
    inset = int(round(forehead_width_inset * tw))
    fx1 = left_x + inset
    fx2 = right_x - inset
    fy2 = brow_y
    fy1 = int(round(brow_y - forehead_height_ratio * face_h))

    fx1, fy1, fx2, fy2 = clamp_rect(fx1, fy1, fx2, fy2, w, h)
    masks["forehead"] = rect_mask(h, w, fx1, fy1, fx2, fy2)

    # ---------- Cheek clamps ----------
    nose_left_limit = int(nose[0] - nose_margin_px)
    nose_right_limit = int(nose[0] + nose_margin_px)

    mouth_y = int(min(mouth_l[1], mouth_r[1]) - mouth_margin_px)
    jaw_y = int(chin_y - jaw_margin_px)
    bottom_cap = int(round(mouth_y + cheek_bottom_frac * max(0, (jaw_y - mouth_y))))

    # ---------- Left cheek base rect ----------
    left_idx = [i for i in LEFT_CHEEK_INDICES if i < len(pts)]
    left_rect = None
    if left_idx:
        x, y, rw, rh = cv2.boundingRect(pts[left_idx])
        x1, y1, x2, y2 = shrink_rect(x, y, rw, rh, shrink=cheek_shrink)

        x2 = min(x2, nose_left_limit)
        y2 = min(y2, mouth_y)
        y2 = min(y2, bottom_cap)

        x1, y1, x2, y2 = clamp_rect(x1, y1, x2, y2, w, h)
        left_rect = (x1, y1, x2, y2)

    # ---------- Right cheek base rect ----------
    right_idx = [i for i in RIGHT_CHEEK_INDICES if i < len(pts)]
    right_rect = None
    if right_idx:
        x, y, rw, rh = cv2.boundingRect(pts[right_idx])
        x1, y1, x2, y2 = shrink_rect(x, y, rw, rh, shrink=cheek_shrink)

        x1 = max(x1, nose_right_limit)
        y2 = min(y2, mouth_y)
        y2 = min(y2, bottom_cap)

        x1, y1, x2, y2 = clamp_rect(x1, y1, x2, y2, w, h)
        right_rect = (x1, y1, x2, y2)

    # Create 2x2 patches + aggregate union mask
    if left_rect is not None:
        x1,y1,x2,y2 = left_rect
        patches = split_rect_2x2(x1,y1,x2,y2)
        union = np.zeros((h,w), np.uint8)
        for pid, (ax1,ay1,ax2,ay2) in patches.items():
            m = rect_mask(h,w,ax1,ay1,ax2,ay2)
            masks[f"left_cheek_{pid}"] = m
            union = cv2.bitwise_or(union, m)
        masks["left_cheek"] = union

    if right_rect is not None:
        x1,y1,x2,y2 = right_rect
        patches = split_rect_2x2(x1,y1,x2,y2)
        union = np.zeros((h,w), np.uint8)
        for pid, (ax1,ay1,ax2,ay2) in patches.items():
            m = rect_mask(h,w,ax1,ay1,ax2,ay2)
            masks[f"right_cheek_{pid}"] = m
            union = cv2.bitwise_or(union, m)
        masks["right_cheek"] = union

    return masks


# ---------------------------
# Main
# ---------------------------
def process_video(
    video_path,
    output_dir="data/processed",
    save_video=True,
    save_data=True,
    skin_mode="ycrcb",  # "ycrcb" | "hsv" | "off"
    lower_hsv=(0, 20, 70),
    upper_hsv=(20, 255, 255),
    cr=(133, 173),
    cb=(77, 127),
    forehead_height_ratio=0.28,
    forehead_width_inset=0.08,
    cheek_shrink=0.18,
    cheek_bottom_frac=0.45,
    nose_margin_px=12,
    mouth_margin_px=10,
    jaw_margin_px=18,
    min_pixels=120,
):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem
    out_video = output_dir / f"{stem}_roi.mp4"
    out_npz = output_dir / f"{stem}_ppg_data.npz"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video), fourcc, fps, (width, height))
        print(f"📹 Output video: {out_video}")

    options = FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = FaceLandmarker.create_from_options(options)

    # --- ROI keys we will save ---
    cheek_patch_ids = ["00","01","10","11"]
    roi_keys = ["forehead", "left_cheek", "right_cheek"] \
               + [f"left_cheek_{pid}" for pid in cheek_patch_ids] \
               + [f"right_cheek_{pid}" for pid in cheek_patch_ids]

    # storage
    sig = {k: [] for k in roi_keys}
    roi_pixels_main = []     # (T,3): forehead, left_cheek(agg), right_cheek(agg)
    roi_pixels_patches = []  # (T,9): forehead, L00,L01,L10,L11, R00,R01,R10,R11
    face_ok = []

    last_good_masks = None

    colors = {
        "forehead": (0, 255, 255),
        "left_cheek": (0, 255, 0),
        "right_cheek": (255, 0, 0),
        # patches: lighter shades to see split
        "left_cheek_00": (0, 200, 0),
        "left_cheek_01": (0, 170, 0),
        "left_cheek_10": (0, 140, 0),
        "left_cheek_11": (0, 110, 0),
        "right_cheek_00": (200, 0, 0),
        "right_cheek_01": (170, 0, 0),
        "right_cheek_10": (140, 0, 0),
        "right_cheek_11": (110, 0, 0),
    }

    frame_idx = 0
    print(f"Processing {video_path.name} ... fps={fps:.3f} frames≈{total_frames}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int(frame_idx * 1000 / fps)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        det = landmarker.detect_for_video(mp_image, timestamp_ms)

        overlay = frame.copy()
        got_face = bool(det.face_landmarks)
        face_ok.append(1 if got_face else 0)

        if got_face:
            pts = landmarks_to_pts(det.face_landmarks[0], width, height)
            masks = get_roi_masks_optionC(
                frame, pts,
                forehead_height_ratio=forehead_height_ratio,
                forehead_width_inset=forehead_width_inset,
                cheek_shrink=cheek_shrink,
                cheek_bottom_frac=cheek_bottom_frac,
                nose_margin_px=nose_margin_px,
                mouth_margin_px=mouth_margin_px,
                jaw_margin_px=jaw_margin_px,
            )
            last_good_masks = masks
        else:
            masks = last_good_masks

        # default values (NaNs)
        frame_means = {k: (np.nan, np.nan, np.nan) for k in roi_keys}
        frame_counts = {k: 0 for k in roi_keys}

        if masks is not None:
            # skin mask
            skin = None
            if skin_mode == "hsv":
                skin = create_skin_mask_hsv(frame, lower_hsv, upper_hsv)
            elif skin_mode == "ycrcb":
                skin = create_skin_mask_ycrcb(frame, cr=cr, cb=cb)
            elif skin_mode == "off":
                skin = None
            else:
                raise ValueError("skin_mode must be: ycrcb | hsv | off")

            # compute patch means
            for k in roi_keys:
                m = masks.get(k, None)
                if m is None:
                    continue
                m2 = cv2.bitwise_and(m, skin) if skin is not None else m
                mean_rgb, cnt = mean_rgb_in_mask(frame, m2)
                if cnt < min_pixels:
                    mean_rgb = (np.nan, np.nan, np.nan)
                frame_means[k] = mean_rgb
                frame_counts[k] = cnt

                # overlay (visualize only if saving video)
                overlay[m2 > 0] = colors.get(k, (255, 255, 255))

            # aggregate cheeks from patches (NaN-safe mean)
            left_patch_means = [frame_means[f"left_cheek_{pid}"] for pid in cheek_patch_ids]
            right_patch_means = [frame_means[f"right_cheek_{pid}"] for pid in cheek_patch_ids]

            if "left_cheek" in masks:
                frame_means["left_cheek"] = nanmean_rgb(left_patch_means)
            if "right_cheek" in masks:
                frame_means["right_cheek"] = nanmean_rgb(right_patch_means)

        annotated = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
        if writer is not None:
            writer.write(annotated)

        # store signals
        for k in roi_keys:
            sig[k].append(frame_means[k])

        # store pixel counts
        roi_pixels_main.append([
            frame_counts.get("forehead", 0),
            frame_counts.get("left_cheek", 0),
            frame_counts.get("right_cheek", 0),
        ])
        roi_pixels_patches.append([
            frame_counts.get("forehead", 0),
            frame_counts.get("left_cheek_00", 0),
            frame_counts.get("left_cheek_01", 0),
            frame_counts.get("left_cheek_10", 0),
            frame_counts.get("left_cheek_11", 0),
            frame_counts.get("right_cheek_00", 0),
            frame_counts.get("right_cheek_01", 0),
            frame_counts.get("right_cheek_10", 0),
            frame_counts.get("right_cheek_11", 0),
        ])

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{total_frames}")

    cap.release()
    if writer is not None:
        writer.release()
    landmarker.close()

    print(f"✅ Finished: {frame_idx} frames")

    if save_data:
        # to arrays
        out_arrays = {k: np.array(sig[k], dtype=np.float32) for k in roi_keys}

        np.savez_compressed(
            out_npz,
            **out_arrays,
            fps=float(fps),
            frame_count=int(frame_idx),
            timestamps=np.arange(frame_idx, dtype=np.float32) / float(fps),
            roi_pixels=np.array(roi_pixels_main, dtype=np.uint32),
            roi_pixels_patches=np.array(roi_pixels_patches, dtype=np.uint32),
            face_ok=np.array(face_ok, dtype=np.uint8),
            meta=dict(
                option="C_multi_patch_2x2",
                skin_mode=skin_mode,
                lower_hsv=tuple(lower_hsv),
                upper_hsv=tuple(upper_hsv),
                cr=tuple(cr),
                cb=tuple(cb),
                forehead_height_ratio=float(forehead_height_ratio),
                forehead_width_inset=float(forehead_width_inset),
                cheek_shrink=float(cheek_shrink),
                cheek_bottom_frac=float(cheek_bottom_frac),
                nose_margin_px=int(nose_margin_px),
                mouth_margin_px=int(mouth_margin_px),
                jaw_margin_px=int(jaw_margin_px),
                min_pixels=int(min_pixels),
                patch_ids=cheek_patch_ids,
            ),
        )

        print(f"📊 PPG data saved to: {out_npz}")
        print("Saved ROI keys:", ", ".join([k for k in roi_keys if k in out_arrays]))


def parse_hsv_triplet(s):
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError("HSV triplet must be like: 0,20,70")
    return tuple(parts)

def parse_int_pair(s):
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 2:
        raise ValueError("Pair must be like: 133,173")
    return tuple(parts)


if __name__ == "__main__":
    ensure_model()

    p = argparse.ArgumentParser()
    p.add_argument("video", nargs="?", default="data/raw/1.MOV")
    p.add_argument("--output", default="data/processed")
    p.add_argument("--no-video", action="store_false", dest="save_video")
    p.add_argument("--no-data", action="store_false", dest="save_data")

    p.add_argument("--skin", choices=["ycrcb", "hsv", "off"], default="ycrcb")

    p.add_argument("--lower-hsv", type=parse_hsv_triplet, default="0,20,70")
    p.add_argument("--upper-hsv", type=parse_hsv_triplet, default="20,255,255")
    p.add_argument("--cr", type=parse_int_pair, default="133,173")
    p.add_argument("--cb", type=parse_int_pair, default="77,127")

    p.add_argument("--forehead-height", type=float, default=0.28)
    p.add_argument("--forehead-inset", type=float, default=0.08)

    p.add_argument("--cheek-shrink", type=float, default=0.18)
    p.add_argument("--cheek-bottom-frac", type=float, default=0.45)

    p.add_argument("--nose-margin", type=int, default=12)
    p.add_argument("--mouth-margin", type=int, default=10)
    p.add_argument("--jaw-margin", type=int, default=18)

    p.add_argument("--min-pixels", type=int, default=120)

    args = p.parse_args()

    process_video(
        args.video,
        output_dir=args.output,
        save_video=args.save_video,
        save_data=args.save_data,
        skin_mode=args.skin,
        lower_hsv=args.lower_hsv,
        upper_hsv=args.upper_hsv,
        cr=args.cr,
        cb=args.cb,
        forehead_height_ratio=args.forehead_height,
        forehead_width_inset=args.forehead_inset,
        cheek_shrink=args.cheek_shrink,
        cheek_bottom_frac=args.cheek_bottom_frac,
        nose_margin_px=args.nose_margin,
        mouth_margin_px=args.mouth_margin,
        jaw_margin_px=args.jaw_margin,
        min_pixels=args.min_pixels,
    )