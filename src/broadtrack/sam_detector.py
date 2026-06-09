"""
SAM3 video segmentation for cricket pitch — demo / preprocessing.

BroadTrack integration (`broadtrack_cricket.py`):
  - `broadtrack_cricket.py --use-lines` expects a *single* pitch mask per frame,
    same shape as YOLO seg: one dominant region for `extract_pitch_edges_from_mask`.
  - This script now picks **one** mask per frame (`--mask-strategy`) instead of
    OR-combining all masks (which drew every adjacent pitch strip).
  - Full tracker integration: either (1) keep YOLO `pitch_seg_best.pt` for speed,
    or (2) add a code path that feeds `select_single_mask(...)` output into the
    same EMA + edge extraction as `--seg-model` (SAM3 is heavier; best as
    offline mask video or batched frames).
"""
import cv2
import numpy as np
import argparse
import os
from typing import Optional

from ultralytics.models.sam import SAM3VideoSemanticPredictor


def _resize_mask_to_hw(mask_np: np.ndarray, h: int, w: int) -> np.ndarray:
    """Binary mask (H,W) float or uint8, resized to (h,w)."""
    if mask_np.ndim > 2:
        mask_np = np.squeeze(mask_np)
    if mask_np.shape[0] != h or mask_np.shape[1] != w:
        mask_np = cv2.resize(mask_np.astype(np.float32), (w, h))
    return mask_np


def score_mask_for_main_pitch(
    mask_bin: np.ndarray,
    h: int,
    w: int,
    anchor_x_norm: float = 0.5,
    anchor_y_norm: float = 0.5,
    spatial_weight: float = 0.22,
) -> float:
    """
    Prefer one pitch: large area + centroid near a *preference point* (not always frame center).

    On multi-pitch squares the **geometric center** often lies on an **empty middle strip**;
    use anchor_x_norm < 0.5 (e.g. `--prefer-side left`) to bias toward the strip where play is.

    anchor_*_norm: 0–1 relative to frame width/height.
    spatial_weight: how much centroid distance matters vs raw area (lower = area wins more).
    """
    ys, xs = np.where(mask_bin > 0.5)
    if len(xs) == 0:
        return -1e18
    area = float(np.sum(mask_bin > 0.5))
    cx, cy = xs.mean(), ys.mean()
    ax, ay = anchor_x_norm * w, anchor_y_norm * h
    dist = np.hypot(cx - ax, cy - ay)
    max_dist = np.hypot(w, h) / 2.0
    spatial_term = 1.0 - (dist / (max_dist + 1e-6))
    # area drives selection; spatial term only breaks ties between similar-sized strips
    return area * ((1.0 - spatial_weight) + spatial_weight * spatial_term)


def select_single_mask(
    masks_data,
    h: int,
    w: int,
    strategy: str = "score",
    anchor_x_norm: float = 0.5,
    anchor_y_norm: float = 0.5,
    spatial_weight: float = 0.22,
) -> Optional[np.ndarray]:
    """
    Pick ONE mask from SAM output instead of OR-combining all (which shows every pitch strip).

    strategies:
      - score: area + centroid near (anchor_x_norm, anchor_y_norm)
      - largest: max area only (ignores anchor)
      - center: closest centroid to the same anchor point
    """
    if masks_data is None or len(masks_data) == 0:
        return None

    tensors = [masks_data[i] for i in range(len(masks_data))]
    candidates = []
    for t in tensors:
        m = t.cpu().numpy()
        m = _resize_mask_to_hw(m, h, w)
        bin_m = (m > 0.5).astype(np.uint8)
        if np.sum(bin_m) < 50:  # ignore specks
            continue
        candidates.append((m, bin_m))

    if not candidates:
        return None

    if strategy == "largest":
        best = max(candidates, key=lambda x: np.sum(x[1]))
        return best[0]

    if strategy == "center":
        ax, ay = anchor_x_norm * w, anchor_y_norm * h

        def cent_dist(item):
            m, bin_m = item
            ys, xs = np.where(bin_m > 0)
            if len(xs) == 0:
                return 1e18
            return np.hypot(xs.mean() - ax, ys.mean() - ay)

        best = min(candidates, key=cent_dist)
        return best[0]

    # score (default)
    best_score = -1e19
    best_m = None
    for m, bin_m in candidates:
        s = score_mask_for_main_pitch(
            bin_m.astype(np.float32),
            h,
            w,
            anchor_x_norm=anchor_x_norm,
            anchor_y_norm=anchor_y_norm,
            spatial_weight=spatial_weight,
        )
        if s > best_score:
            best_score = s
            best_m = m
    return best_m


def main():
    parser = argparse.ArgumentParser(description="SAM 3 Cricket Pitch Video Segmentation")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Path to output video")
    parser.add_argument("--text", default="main cricket pitch center strip", help="Text prompt for the concept to segment")
    parser.add_argument("--use-bbox", action="store_true", help="Use a manually drawn bounding box on frame 1 instead of text")
    parser.add_argument("--device", default="cpu", help="Device to run on (cpu or cuda)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument(
        "--mask-strategy",
        default="score",
        choices=("score", "largest", "center"),
        help="How to pick ONE pitch when SAM returns multiple masks (default: score = area + center bias)",
    )
    parser.add_argument("--save-masks", type=str, default=None,
                        help="Directory to save raw binary mask PNGs per frame (for broadtrack_cricket.py --sam-masks)")
    args = parser.parse_args()

    # Create mask output directory if requested
    if args.save_masks:
        os.makedirs(args.save_masks, exist_ok=True)
        print(f"Will save binary masks to: {args.save_masks}/")

    # --- Optional: Get bounding box from first frame ---
    bboxes = None
    if args.use_bbox:
        cap = cv2.VideoCapture(args.input)
        ret, first_frame = cap.read()
        cap.release()
        if not ret:
            print("Failed to read first frame for bbox selection.")
            return

        print("Draw a box TIGHTLY around the pitch area, then press ENTER/SPACE.")
        roi = cv2.selectROI("Draw Box Around Pitch", first_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Draw Box Around Pitch")
        x, y, w, h = roi
        bboxes = [[x, y, x + w, y + h]]
        print(f"Using bounding box: {bboxes[0]}")

    # --- Initialize SAM 3 Video Semantic Predictor ---
    print(f"Initializing SAM 3 Video Predictor (device={args.device})...")
    overrides = dict(
        conf=args.conf,
        task="segment",
        mode="predict",
        model="sam3.pt",
        half=(args.device != "cpu"),  # FP16 only on GPU
        save=False,
        device=args.device,
    )
    predictor = SAM3VideoSemanticPredictor(overrides=overrides)

    # --- Run video tracking ---
    print(f"Running SAM 3 video tracking on: {args.input}")
    if args.use_bbox and bboxes:
        print(f"  Prompt: bounding box {bboxes[0]}")
        results = predictor(
            source=args.input,
            bboxes=bboxes,
            labels=[1],  # positive label
            stream=True,
        )
    else:
        print(f"  Prompt: text='{args.text}'")
        results = predictor(
            source=args.input,
            text=[args.text],
            stream=True,
        )

    # --- Process results and write output video ---
    out = None
    frame_count = 0
    logged_multi = False
    
    # Temporal tracking: start anchored at the center of the frame
    current_anchor_x = 0.5
    current_anchor_y = 0.5

    for r in results:
        # Get the original frame
        frame = r.orig_img.copy()
        h, w = frame.shape[:2]

        # Initialize video writer on first frame
        if out is None:
            fps = 30  # Default; will be overridden if we can read from source
            cap_tmp = cv2.VideoCapture(args.input)
            if cap_tmp.isOpened():
                fps = int(cap_tmp.get(cv2.CAP_PROP_FPS)) or 30
                cap_tmp.release()
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
            print(f"Output video: {args.output} ({w}x{h} @ {fps}fps)")

        # Overlay mask if available — pick ONE mask
        if r.masks is not None and len(r.masks.data) > 0:
            if not logged_multi and len(r.masks.data) > 1:
                print(
                    f"  [INFO] SAM returned {len(r.masks.data)} masks/frame — "
                    f"keeping one via --mask-strategy={args.mask_strategy} (with Dynamic Temporal Tracking)"
                )
                logged_multi = True
                
            best = select_single_mask(
                r.masks.data, h, w, strategy=args.mask_strategy,
                anchor_x_norm=current_anchor_x,
                anchor_y_norm=current_anchor_y,
                spatial_weight=0.85  # Increase spatial weight to heavily favor staying near the tracked centroid
            )
            
            if best is not None:
                if best.ndim > 2:
                    best = np.squeeze(best)
                
                # Get the raw, jagged mask from SAM
                raw_mask = (best > 0.5).astype(np.uint8) * 255

                # --- 1. THE SPATIAL FILTER (Cookie Cutter) ---
                # This guarantees the mask CANNOT drift to the practice pitch
                if args.use_bbox and bboxes:
                    orig_x1, orig_y1, orig_x2, orig_y2 = bboxes[0]
                    
                    # Add a 40-pixel safety buffer for slight camera vibrations
                    buffer = 40
                    x1 = max(0, int(orig_x1) - buffer)
                    y1 = max(0, int(orig_y1) - buffer)
                    x2 = min(w, int(orig_x2) + buffer)
                    y2 = min(h, int(orig_y2) + buffer)

                    # Create a blank canvas and draw a white box over our allowed area
                    allow_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.rectangle(allow_mask, (x1, y1), (x2, y2), 255, -1)

                    # Delete any green pixels that SAM generated outside our allowed area
                    raw_mask = cv2.bitwise_and(raw_mask, allow_mask)
                # ------------------------------------------

                # --- 2. THE GEOMETRY FIX (Perspective-accurate Convex Hull) ---
                contours, _ = cv2.findContours(
                    raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                # Create a fresh canvas for our solid shape
                combined_mask = np.zeros((h, w), dtype=np.uint8)
                
                if contours:
                    # Group all the fragmented green pieces of the pitch together
                    all_points = np.vstack(contours)
                    
                    # Calculate the perspective-accurate convex hull
                    hull = cv2.convexHull(all_points)
                    
                    # Draw the perfectly fitted polygon
                    cv2.drawContours(combined_mask, [hull], 0, 255, thickness=cv2.FILLED)
                    
                    # UPDATE TEMPORAL TRACKER
                    # Find the exact center pixel of our final tracking shape
                    M = cv2.moments(hull)
                    if M["m00"] != 0:
                        cx = M["m10"] / M["m00"]
                        cy = M["m01"] / M["m00"]
                        current_anchor_x = cx / w
                        current_anchor_y = cy / h
                # --------------------------------------------------------------

                # Save raw binary mask for BroadTrack integration
                if args.save_masks:
                    mask_path = os.path.join(args.save_masks, f"mask_{frame_count:06d}.png")
                    cv2.imwrite(mask_path, combined_mask)

                # Create the visual green overlay
                overlay = np.zeros_like(frame)
                overlay[combined_mask == 255] = [0, 255, 0]
                frame = cv2.addWeighted(frame, 1.0, overlay, 0.4, 0)

                # Draw the final straight-edged contour outline
                final_contours, _ = cv2.findContours(
                    combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(frame, final_contours, -1, (0, 255, 0), 2)
        else:
            # No mask detected — save empty mask to keep frame numbering consistent
            if args.save_masks:
                mask_path = os.path.join(args.save_masks, f"mask_{frame_count:06d}.png")
                cv2.imwrite(mask_path, np.zeros((h, w), dtype=np.uint8))

        out.write(frame)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  Processed {frame_count} frames...")

    if out is not None:
        out.release()

    print(f"\n[DONE] Processed {frame_count} frames → {args.output}")


if __name__ == "__main__":
    main()