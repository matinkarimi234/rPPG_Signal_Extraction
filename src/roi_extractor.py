"""
Extract ROI mean RGB signals for rPPG and save:
1) annotated video with ROI overlays
2) .npz with RGB means per ROI + timestamps + quality stats
Headless, no GUI.

ROIs:
- forehead: rectangle between temples above brows
- left_cheek/right_cheek: bounding rectangles from landmark clusters (optionally shrunk)

Output NPZ keys:
- forehead, left_cheek, right_cheek: (T,3) float32 in RGB order
- fps, frame_count, timestamps
- roi_pixels: (T,3) uint32 pixel-counts for [forehead,left,right]
- face_ok: (T,) uint8 face detected flag
"""

import os
import sys
import cv2
import numpy as np
import urllib.request
from pathlib import Path
import argparse

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode


# ---------------------------
# Model download
# ---------------------------
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = "face_landmarker.task"


def ensure_model(path=MODEL_PATH, url=MODEL_URL):
    if not os.path.exists(path):
        print("Downloading face_landmarker.task...")
        urllib.request.urlretrieve(url, path)
        print("Download complete.")


# ---------------------------
# Landmark sets (MediaPipe Face Mesh)
# ---------------------------
BROW_INDICES = [46, 53, 52, 51, 48, 276, 283, 282, 281, 278]
LEFT_CHEEK_INDICES = [118, 119, 100, 136, 206, 205, 49, 203, 227, 137, 177, 215, 216]
RIGHT_CHEEK_INDICES = [346, 345, 374, 423, 424, 432, 279, 422, 426, 436, 434, 416, 376]
# Temples for forehead width:
LEFT_TEMPLE = 127
RIGHT_TEMPLE = 356


# ---------------------------
# Skin mask (optional)
# ---------------------------
def create_skin_mask_bgr(image_bgr, lower_hsv=(0, 20, 70), upper_hsv=(20, 255, 255)):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
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
    if x2 <= x1: x2 = min(w - 1, x1 + 1)
    if y2 <= y1: y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def rect_mask(h, w, x1, y1, x2, y2):
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(m, (x1, y1), (x2, y2), 255, -1)
    return m


def shrink_rect(x, y, rw, rh, shrink=0.10):
    """
    shrink=0.10 means 10% border removed on each side.
    """
    dx = int(round(rw * shrink))
    dy = int(round(rh * shrink))
    x2 = x + rw
    y2 = y + rh
    return x + dx, y + dy, x2 - dx, y2 - dy


def landmarks_to_pts(landmarks, w, h):
    return np.array([(int(lm.x * w), int(lm.y * h)) for lm in landmarks], dtype=np.int32)


def get_roi_masks(frame_bgr, pts, forehead_expand=0.10, cheek_shrink=0.10):
    """
    Return dict of {roi_name: mask_uint8}.
    forehead_expand expands forehead height upward slightly (ratio of forehead height).
    cheek_shrink shrinks cheek rectangles to reduce contamination.
    """
    h, w = frame_bgr.shape[:2]
    masks = {}

    # Forehead:
    top_y = int(np.min(pts[:, 1]))
    brow_y = int(np.min(pts[BROW_INDICES, 1])) if np.max(BROW_INDICES) < len(pts) else top_y + int(0.2 * h)

    left_x = int(pts[LEFT_TEMPLE, 0]) if LEFT_TEMPLE < len(pts) else int(0.3 * w)
    right_x = int(pts[RIGHT_TEMPLE, 0]) if RIGHT_TEMPLE < len(pts) else int(0.7 * w)

    # Expand forehead upward a bit (useful if top_y is “too high” due to hairline jitter)
    fh = max(2, brow_y - top_y)
    top_y2 = top_y - int(round(forehead_expand * fh))

    x1, y1, x2, y2 = clamp_rect(left_x, top_y2, right_x, brow_y, w, h)
    masks["forehead"] = rect_mask(h, w, x1, y1, x2, y2)

    # Left cheek:
    left_idx = [i for i in LEFT_CHEEK_INDICES if i < len(pts)]
    if left_idx:
        x, y, rw, rh = cv2.boundingRect(pts[left_idx])
        x1, y1, x2, y2 = shrink_rect(x, y, rw, rh, shrink=cheek_shrink)
        x1, y1, x2, y2 = clamp_rect(x1, y1, x2, y2, w, h)
        masks["left_cheek"] = rect_mask(h, w, x1, y1, x2, y2)

    # Right cheek:
    right_idx = [i for i in RIGHT_CHEEK_INDICES if i < len(pts)]
    if right_idx:
        x, y, rw, rh = cv2.boundingRect(pts[right_idx])
        x1, y1, x2, y2 = shrink_rect(x, y, rw, rh, shrink=cheek_shrink)
        x1, y1, x2, y2 = clamp_rect(x1, y1, x2, y2, w, h)
        masks["right_cheek"] = rect_mask(h, w, x1, y1, x2, y2)

    return masks


def mean_rgb_in_mask(frame_bgr, mask):
    """
    Returns (R,G,B) float32, and pixel_count.
    """
    cnt = int(np.count_nonzero(mask))
    if cnt < 20:
        return (np.nan, np.nan, np.nan), cnt
    b, g, r, _ = cv2.mean(frame_bgr, mask=mask)
    return (float(r), float(g), float(b)), cnt


# ---------------------------
# Main
# ---------------------------
def process_video(
    video_path,
    output_dir="data/processed",
    save_video=True,
    save_data=True,
    use_skin_mask=True,
    lower_hsv=(0, 20, 70),
    upper_hsv=(20, 255, 255),
    forehead_expand=0.10,
    cheek_shrink=0.10,
    min_pixels=80,
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

    # outputs
    roi_names = ["forehead", "left_cheek", "right_cheek"]
    sig = {k: [] for k in roi_names}
    roi_pixels = []
    face_ok = []

    last_good_masks = None  # fallback if detection fails

    colors = {"forehead": (0, 255, 255), "left_cheek": (0, 255, 0), "right_cheek": (255, 0, 0)}

    frame_idx = 0
    print(f"Processing {video_path.name} ... fps={fps:.3f} frames≈{total_frames}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int(frame_idx * 1000 / fps)

        # MediaPipe wants RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        det = landmarker.detect_for_video(mp_image, timestamp_ms)

        overlay = frame.copy()
        frame_means = {}
        frame_counts = {k: 0 for k in roi_names}

        got_face = bool(det.face_landmarks)
        face_ok.append(1 if got_face else 0)

        if got_face:
            pts = landmarks_to_pts(det.face_landmarks[0], width, height)
            masks = get_roi_masks(
                frame,
                pts,
                forehead_expand=forehead_expand,
                cheek_shrink=cheek_shrink,
            )
            last_good_masks = masks
        else:
            masks = last_good_masks  # fallback

        if masks is not None:
            skin = None
            if use_skin_mask:
                skin = create_skin_mask_bgr(frame, lower_hsv, upper_hsv)

            for name in roi_names:
                m = masks.get(name, None) if masks else None
                if m is None:
                    frame_means[name] = (np.nan, np.nan, np.nan)
                    frame_counts[name] = 0
                    continue

                if skin is not None:
                    m2 = cv2.bitwise_and(m, skin)
                else:
                    m2 = m

                mean_rgb, cnt = mean_rgb_in_mask(frame, m2)
                # enforce a minimum pixel count for quality
                if cnt < min_pixels:
                    mean_rgb = (np.nan, np.nan, np.nan)

                frame_means[name] = mean_rgb
                frame_counts[name] = cnt

                # overlay
                overlay[m2 > 0] = colors.get(name, (255, 255, 255))

        # blend
        annotated = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

        if writer is not None:
            writer.write(annotated)

        for name in roi_names:
            sig[name].append(frame_means.get(name, (np.nan, np.nan, np.nan)))

        roi_pixels.append([frame_counts["forehead"], frame_counts["left_cheek"], frame_counts["right_cheek"]])

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{total_frames}")

    cap.release()
    if writer is not None:
        writer.release()
    landmarker.close()

    print(f"✅ Finished: {frame_idx} frames")

    if save_data:
        forehead = np.array(sig["forehead"], dtype=np.float32)
        left = np.array(sig["left_cheek"], dtype=np.float32)
        right = np.array(sig["right_cheek"], dtype=np.float32)

        np.savez_compressed(
            out_npz,
            forehead=forehead,
            left_cheek=left,
            right_cheek=right,
            fps=float(fps),
            frame_count=int(frame_idx),
            timestamps=np.arange(frame_idx, dtype=np.float32) / float(fps),
            roi_pixels=np.array(roi_pixels, dtype=np.uint32),
            face_ok=np.array(face_ok, dtype=np.uint8),
            meta=dict(
                use_skin_mask=bool(use_skin_mask),
                lower_hsv=tuple(lower_hsv),
                upper_hsv=tuple(upper_hsv),
                forehead_expand=float(forehead_expand),
                cheek_shrink=float(cheek_shrink),
                min_pixels=int(min_pixels),
            ),
        )

        print(f"📊 PPG data saved to: {out_npz}")
        print("    forehead:", forehead.shape, "left:", left.shape, "right:", right.shape)


def parse_hsv_triplet(s):
    # "0,20,70" -> (0,20,70)
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError("HSV triplet must be like: 0,20,70")
    return tuple(parts)


if __name__ == "__main__":
    ensure_model()

    p = argparse.ArgumentParser()
    p.add_argument("video", nargs="?", default="data/raw/1.MOV")
    p.add_argument("--output", default="data/processed")
    p.add_argument("--no-video", action="store_false", dest="save_video")
    p.add_argument("--no-data", action="store_false", dest="save_data")

    p.add_argument("--no-skin", action="store_false", dest="use_skin_mask",
                   help="Disable HSV skin masking (keep geometric ROI only).")
    p.add_argument("--lower-hsv", type=parse_hsv_triplet, default="0,20,70",
                   help="Lower HSV bound, e.g. 0,20,70")
    p.add_argument("--upper-hsv", type=parse_hsv_triplet, default="20,255,255",
                   help="Upper HSV bound, e.g. 20,255,255")

    p.add_argument("--forehead-expand", type=float, default=0.10,
                   help="Expand forehead upward by this ratio of forehead height.")
    p.add_argument("--cheek-shrink", type=float, default=0.10,
                   help="Shrink cheek rectangles by this ratio to reduce contamination.")
    p.add_argument("--min-pixels", type=int, default=80,
                   help="Minimum ROI pixels required; otherwise write NaNs for that frame/ROI.")

    # robust fallback for weird consoles
    try:
        args = p.parse_args()
    except SystemExit:
        args = argparse.Namespace(
            video="data/raw/454.MOV",
            output="data/processed",
            save_video=True,
            save_data=True,
            use_skin_mask=True,
            lower_hsv=(0, 20, 70),
            upper_hsv=(20, 255, 255),
            forehead_expand=0.10,
            cheek_shrink=0.10,
            min_pixels=80,
        )

    process_video(
        args.video,
        output_dir=args.output,
        save_video=args.save_video,
        save_data=args.save_data,
        use_skin_mask=args.use_skin_mask,
        lower_hsv=args.lower_hsv,
        upper_hsv=args.upper_hsv,
        forehead_expand=args.forehead_expand,
        cheek_shrink=args.cheek_shrink,
        min_pixels=args.min_pixels,
    )
