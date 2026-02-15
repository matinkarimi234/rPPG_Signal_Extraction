"""
Extract PPG signals and save annotated video using MediaPipe FaceLandmarker.
- Input : video file
- Output: 
   1) processed video with ROI overlays (saved to data/processed/*_roi.mp4)
   2) .npz file with RGB means for each ROI (saved to same folder)
No GUI, fully headless.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
import os
import urllib.request
from pathlib import Path
import argparse
import sys

# ------------------------------------------------------------
# 1. Download model if not present
# ------------------------------------------------------------
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = "face_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading face_landmarker.task...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

# ------------------------------------------------------------
# 2. Landmark indices (478-point model)
# ------------------------------------------------------------
FOREHEAD_INDICES = [10, 67, 54, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 349]
LEFT_CHEEK_INDICES = [118, 119, 100, 136, 206, 205, 49, 203, 227, 137, 177, 215, 216]
RIGHT_CHEEK_INDICES = [346, 345, 374, 423, 424, 432, 279, 422, 426, 436, 434, 416, 376]

# ------------------------------------------------------------
# 3. Skin segmentation utility
# ------------------------------------------------------------
def create_skin_mask(image, lower_hsv=(0, 20, 70), upper_hsv=(20, 255, 255)):
    """HSV‑based skin detector – returns binary mask."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

# ------------------------------------------------------------
# 4. ROI mask generation from landmarks
# ------------------------------------------------------------
def get_roi_masks(image, landmarks, expand_forehead_ratio=0.0):
    """
    Generate clean rectangular ROIs:
    - Forehead : rectangle from top of head to eyebrow line, between temples.
    - Cheeks   : bounding rectangles of cheek landmark clusters.
    """
    h, w = image.shape[:2]
    masks = {}

    pts = np.array([(int(lm.x * w), int(lm.y * h)) for lm in landmarks])

    # --------------------------------------------------------
    # 1. Forehead ROI (rectangle between temples, above brows)
    # --------------------------------------------------------
    # Top of head = minimum y coordinate
    top_y = np.min(pts[:, 1])
    
    # Eyebrow ridge: use the highest (smallest y) among brow points
    brow_indices = [46, 53, 52, 51, 48, 276, 283, 282, 281, 278]
    brow_y = np.min(pts[brow_indices, 1]) if all(i < len(pts) for i in brow_indices) else top_y + int(h * 0.2)
    
    # Temple points (left: 127, right: 356) for forehead width
    left_temple_x = pts[127][0] if 127 < len(pts) else int(w * 0.3)
    right_temple_x = pts[356][0] if 356 < len(pts) else int(w * 0.7)
    
    forehead_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(forehead_mask, 
                  (left_temple_x, top_y), 
                  (right_temple_x, brow_y), 
                  255, -1)
    masks['forehead'] = forehead_mask

    # --------------------------------------------------------
    # 2. Cheek ROIs – **RECTANGLES** instead of convex hulls
    # --------------------------------------------------------
    # Left cheek landmarks
    left_cheek_indices = [118, 119, 100, 136, 206, 205, 49, 203, 227, 137, 177, 215, 216]
    left_pts = pts[[i for i in left_cheek_indices if i < len(pts)]]
    if len(left_pts) > 0:
        x, y, w_rect, h_rect = cv2.boundingRect(left_pts)
        left_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(left_mask, (x, y), (x + w_rect, y + h_rect), 255, -1)
        masks['left_cheek'] = left_mask

    # Right cheek landmarks
    right_cheek_indices = [346, 345, 374, 423, 424, 432, 279, 422, 426, 436, 434, 416, 376]
    right_pts = pts[[i for i in right_cheek_indices if i < len(pts)]]
    if len(right_pts) > 0:
        x, y, w_rect, h_rect = cv2.boundingRect(right_pts)
        right_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(right_mask, (x, y), (x + w_rect, y + h_rect), 255, -1)
        masks['right_cheek'] = right_mask

    return masks

# ------------------------------------------------------------
# 5. Main processing function
# ------------------------------------------------------------
def process_video(video_path, output_dir="data/processed", save_video=True, save_data=True):
    """
    Process video: detect ROIs, draw overlays, write to output video,
    and collect RGB mean time series.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filenames
    stem = video_path.stem
    output_video_path = output_dir / f"{stem}_roi.mp4"
    output_data_path = output_dir / f"{stem}_ppg_data.npz"

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialise VideoWriter if saving video
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        print(f"📹 Output video: {output_video_path}")

    # Initialise FaceLandmarker
    options = FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    landmarker = FaceLandmarker.create_from_options(options)

    # Data storage for PPG signals
    signals = {
        'forehead': [],
        'left_cheek': [],
        'right_cheek': []
    }

    frame_idx = 0
    print(f"Processing {video_path.name} ...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Timestamp in milliseconds (required by detect_for_video)
        timestamp_ms = int(frame_idx * 1000 / fps)

        # Convert to MediaPipe Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Detect landmarks
        detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Initialise overlay frame as copy of original
        overlay = frame.copy()
        rgb_means = {}  # store means for this frame

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]

            # Generate geometric masks
            roi_masks = get_roi_masks(frame, landmarks)

            # Skin refinement
            skin_mask = create_skin_mask(frame)

            # Draw refined ROIs and compute means
            colors = {'forehead': (0, 255, 255), 'left_cheek': (0, 255, 0), 'right_cheek': (255, 0, 0)}

            for name, mask in roi_masks.items():
                if mask is None:
                    continue
                # Refine mask with skin segmentation
                refined_mask = cv2.bitwise_and(mask, skin_mask)

                # Compute mean RGB in the refined region
                if np.count_nonzero(refined_mask) > 10:  # avoid tiny regions
                    mean_rgb = cv2.mean(frame, mask=refined_mask)[:3]  # (B, G, R) in OpenCV order
                    # Store as (R, G, B) for consistency
                    rgb_means[name] = (mean_rgb[2], mean_rgb[1], mean_rgb[0])
                else:
                    rgb_means[name] = (np.nan, np.nan, np.nan)

                # Draw the refined mask as a coloured overlay (semi‑transparent)
                overlay[refined_mask > 0] = colors.get(name, (255, 255, 255))

        # Blend overlay with original frame (alpha = 0.4)
        annotated_frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

        # Write frame to output video
        if save_video and writer is not None:
            writer.write(annotated_frame)

        # Append data to signals (use NaN if no face or no ROI)
        for roi in ['forehead', 'left_cheek', 'right_cheek']:
            if roi in rgb_means:
                signals[roi].append(rgb_means[roi])
            else:
                signals[roi].append((np.nan, np.nan, np.nan))

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames")

    # Cleanup
    cap.release()
    if writer is not None:
        writer.release()
    landmarker.close()

    print(f"✅ Finished processing {frame_idx} frames.")

    # --------------------------------------------------------
    # 6. Save PPG data
    # --------------------------------------------------------
    if save_data:
        # Convert lists to numpy arrays
        forehead_arr = np.array(signals['forehead'], dtype=np.float32)   # shape (T,3)
        left_cheek_arr = np.array(signals['left_cheek'], dtype=np.float32)
        right_cheek_arr = np.array(signals['right_cheek'], dtype=np.float32)

        # Save as .npz
        np.savez_compressed(
            output_data_path,
            forehead=forehead_arr,
            left_cheek=left_cheek_arr,
            right_cheek=right_cheek_arr,
            fps=fps,
            frame_count=frame_idx,
            timestamps=np.arange(frame_idx) / fps
        )
        print(f"📊 PPG data saved to: {output_data_path}")
        print(f"    - Forehead   : {forehead_arr.shape}")
        print(f"    - Left cheek : {left_cheek_arr.shape}")
        print(f"    - Right cheek: {right_cheek_arr.shape}")

# ------------------------------------------------------------
# 7. Entry point with robust argument parsing
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract facial ROIs, save annotated video and PPG time series.")
    parser.add_argument("video", type=str, nargs='?', default="data/raw/454.MOV",
                        help="Path to input video (default: data/raw/508.MOV)")
    parser.add_argument("--output", type=str, default="data/processed",
                        help="Directory to save outputs (default: data/processed)")
    parser.add_argument("--no-video", action="store_false", dest="save_video",
                        help="Disable saving of processed video")
    parser.add_argument("--no-data", action="store_false", dest="save_data",
                        help="Disable saving of PPG .npz data")
    parser.add_argument("--no-vis", action="store_true", help="Deprecated – kept for compatibility")

    # Handle environments without valid console handles (e.g., some IDEs)
    try:
        if sys.stdout is None or sys.stderr is None:
            raise OSError
        args = parser.parse_args()
    except (OSError, AttributeError, SystemExit):
        # Fallback to defaults
        print("⚠️  Console environment not detected – using default arguments.", file=sys.stderr)
        args = argparse.Namespace(
            video="data/raw/508.MOV",
            output="data/processed",
            save_video=True,
            save_data=True
        )

    process_video(args.video, args.output, args.save_video, args.save_data)