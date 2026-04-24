"""
BroadTrack Cricket — Pipeline Evaluation

Answers the question: "Are my camera parameters correct or random?"

Three automated tests:
  1. Physical sanity checks (no ground truth needed)
  2. Reprojection error aggregation
  3. Line IoU — projected pitch outline vs segmentation mask

Usage:
  python evaluate_pipeline.py \
    -j bt_output_A_fixed_v2.json \
    -v input_videos/input1.mp4 \
    -s ../models/pitch_seg_best.pt
"""

import argparse
import json
import sys
import os
import cv2
import numpy as np
from pathlib import Path

from broadtrack.camera import project_points_batch
from broadtrack.camera_tracker_sam import PITCH_CORNERS_3D


# ─────────────────────────────────────────────────────
# TEST 1: PHYSICAL SANITY CHECKS
# ─────────────────────────────────────────────────────

SANITY_BOUNDS = {
    "Z_min": 5.0,
    "Z_max": 50.0,
    "hfov_min": 3.0,
    "hfov_max": 60.0,
    "X_max": 200.0,
    "Y_max": 50.0,
}


def check_physical_sanity(cp):
    """Returns (passed: bool, reasons: list[str]) for a single frame."""
    reasons = []
    z = cp.get("positionZMeters")
    hfov = cp.get("horizontalFieldOfViewDegrees")
    x = cp.get("positionXMeters")
    y = cp.get("positionYMeters")

    if z is not None:
        if z < SANITY_BOUNDS["Z_min"] or z > SANITY_BOUNDS["Z_max"]:
            reasons.append(f"Z={z:.1f}m outside [{SANITY_BOUNDS['Z_min']}, {SANITY_BOUNDS['Z_max']}]")
    if hfov is not None:
        if hfov < SANITY_BOUNDS["hfov_min"] or hfov > SANITY_BOUNDS["hfov_max"]:
            reasons.append(f"HFOV={hfov:.1f}° outside [{SANITY_BOUNDS['hfov_min']}, {SANITY_BOUNDS['hfov_max']}]")
    if x is not None and abs(x) > SANITY_BOUNDS["X_max"]:
        reasons.append(f"|X|={abs(x):.1f}m > {SANITY_BOUNDS['X_max']}")
    if y is not None and abs(y) > SANITY_BOUNDS["Y_max"]:
        reasons.append(f"|Y|={abs(y):.1f}m > {SANITY_BOUNDS['Y_max']}")

    return len(reasons) == 0, reasons


def run_test1_sanity(frames_data):
    """Test 1: Physical sanity checks on all tracked frames."""
    print("\n" + "=" * 60)
    print("TEST 1: PHYSICAL SANITY CHECKS")
    print("=" * 60)

    tracked = [(k, v) for k, v in frames_data.items() if v.get("score", 0) > 0]
    if not tracked:
        print("  No tracked frames found.")
        return 0.0

    passed = 0
    failed_examples = []
    for key, entry in tracked:
        ok, reasons = check_physical_sanity(entry["cp"])
        if ok:
            passed += 1
        elif len(failed_examples) < 5:
            failed_examples.append((key, reasons))

    rate = passed / len(tracked)
    print(f"  Tracked frames:  {len(tracked)}")
    print(f"  Sanity passed:   {passed}/{len(tracked)} ({rate*100:.1f}%)")

    if failed_examples:
        print(f"\n  Sample failures:")
        for key, reasons in failed_examples:
            print(f"    {key}: {'; '.join(reasons)}")

    verdict = "PASS" if rate > 0.9 else ("MARGINAL" if rate > 0.7 else "FAIL")
    print(f"\n  Verdict: {verdict}")
    return rate


# ─────────────────────────────────────────────────────
# TEST 2: REPROJECTION ERROR AGGREGATION
# ─────────────────────────────────────────────────────

def run_test2_reproj(frames_data):
    """Test 2: Aggregate reprojection errors from OPTIMIZED frames."""
    print("\n" + "=" * 60)
    print("TEST 2: REPROJECTION ERROR")
    print("=" * 60)

    errors = []
    for key, entry in frames_data.items():
        re = entry.get("reproj_error_px")
        if re is not None:
            errors.append(re)

    if not errors:
        print("  No reprojection errors recorded in JSON.")
        print("  (Re-run the pipeline with the updated broadtrack_cricket.py to log them.)")
        return None

    errors = np.array(errors)
    mean_e = np.mean(errors)
    median_e = np.median(errors)
    p90_e = np.percentile(errors, 90)

    print(f"  Frames with reproj error: {len(errors)}")
    print(f"  Mean:   {mean_e:.1f} px")
    print(f"  Median: {median_e:.1f} px")
    print(f"  90th %%: {p90_e:.1f} px")

    verdict = "PASS" if mean_e < 15 else ("MARGINAL" if mean_e < 40 else "FAIL")
    print(f"\n  Verdict: {verdict}  (PASS < 15px, MARGINAL < 40px, FAIL >= 40px)")
    return mean_e


# ─────────────────────────────────────────────────────
# TEST 3: LINE IoU (projected pitch vs seg mask)
# ─────────────────────────────────────────────────────

def compute_line_iou(params, seg_mask, image_w, image_h, yaw_flip=False):
    """
    Compute IoU between projected pitch polygon and segmentation mask.
    params: 8-element camera param vector
    seg_mask: binary mask from seg model (H, W), values 0 or 1
    yaw_flip: whether the tracker used flipped orientation for this frame
    Returns IoU float, or None if projection fails.
    """
    pp = np.array([image_w / 2.0, image_h / 2.0])

    outer_corners = PITCH_CORNERS_3D[:4].copy()
    if yaw_flip:
        outer_corners[:, 0:2] *= -1

    projected, valid = project_points_batch(params, outer_corners, pp)

    if not np.all(valid):
        return None

    corners_2d = projected.astype(np.int32)

    proj_mask = np.zeros((image_h, image_w), dtype=np.uint8)
    cv2.fillPoly(proj_mask, [corners_2d], 1)

    intersection = np.sum(proj_mask & seg_mask)
    union = np.sum(proj_mask | seg_mask)

    if union == 0:
        return None

    return float(intersection) / float(union)


def run_test3_line_iou(frames_data, video_path, seg_model_path=None, sam_masks_dir=None, vis_dir=None):
    """Test 3: Line IoU — projected pitch vs segmentation mask."""
    print("\n" + "=" * 60)
    print("TEST 3: LINE IoU (projected pitch vs seg mask)")
    print("=" * 60)

    if not video_path or (not seg_model_path and not sam_masks_dir):
        print("  Skipped — requires --video AND either --seg-model or --sam-masks.")
        return None

    seg_model = None
    if seg_model_path:
        try:
            from ultralytics import YOLO
            seg_model = YOLO(seg_model_path)
        except ImportError:
            print("  Skipped — ultralytics not installed.")
            return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Error: Cannot open video {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create vis directory if requested
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
        print(f"  Saving ALL frames to: {vis_dir}/")

    # Build a lookup: frame_index -> entry
    frame_lookup = {}
    for key, entry in frames_data.items():
        idx = int(entry.get("time", 0))
        frame_lookup[idx] = entry

    # Color palette for status banners (BGR)
    STATUS_COLORS = {
        "OPTIMIZED":    (0, 200, 0),      # Green
        "INIT":         (0, 200, 0),      # Green
        "REINIT":       (0, 200, 0),      # Green
        "FLOW":         (200, 200, 0),    # Cyan
        "COAST":        (0, 140, 255),    # Orange
        "OPT_REJECTED": (0, 0, 220),      # Red
        "NO_MASK":      (0, 200, 200),    # Yellow
        "NO_TRACKER":   (80, 80, 80),     # Gray
    }

    ious = []
    frames_evaluated = 0

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        entry = frame_lookup.get(frame_idx, None)
        mode = entry.get("mode", "") if entry else ""
        cp = entry.get("cp", {}) if entry else {}
        has_rvec = "rvec" in cp
        is_tracked = mode in ("OPTIMIZED", "FLOW", "INIT", "REINIT")

        # --- Build camera params if available ---
        params = None
        yaw_flip = False
        if is_tracked and has_rvec:
            rvec = np.array(cp["rvec"])
            position = np.array([cp["positionXMeters"], cp["positionYMeters"], cp["positionZMeters"]])
            focal = cp["focal"]
            params = np.zeros(8)
            params[0:3] = rvec
            params[3:6] = position
            params[6] = focal
            params[7] = 0.0
            yaw_flip = cp.get("yaw_flip", False)

        # --- Run segmentation (or load precomputed mask) ---
        seg_mask = None
        if params is not None:
            if sam_masks_dir:
                mask_path = os.path.join(sam_masks_dir, f"mask_{frame_idx:06d}.png")
                if os.path.exists(mask_path):
                    raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if raw is not None:
                        mask_resized = cv2.resize(raw, (frame_w, frame_h))
                        seg_mask = (mask_resized > 127).astype(np.uint8)
            elif seg_model is not None:
                seg_results = seg_model(frame, conf=0.25, verbose=False)
                sr = seg_results[0]
                if sr.masks is not None and len(sr.masks.data) > 0:
                    mask_raw = sr.masks.data[0].cpu().numpy()
                    mask_resized = cv2.resize(mask_raw, (frame_w, frame_h))
                    seg_mask = (mask_resized > 0.5).astype(np.uint8)

        # --- Compute IoU if possible ---
        iou = None
        if params is not None and seg_mask is not None:
            iou = compute_line_iou(params, seg_mask, frame_w, frame_h, yaw_flip=yaw_flip)
            if iou is not None:
                ious.append(iou)
                frames_evaluated += 1

        # --- Determine display status ---
        if is_tracked and iou is not None:
            display_mode = mode
            label = f"Frame {frame_idx} | {mode} | IoU: {iou:.3f}"
        elif is_tracked and seg_mask is None:
            display_mode = "NO_MASK"
            label = f"Frame {frame_idx} | {mode} | NO SEG MASK"
        elif mode in ("COAST", "OPT_REJECTED"):
            display_mode = mode
            label = f"Frame {frame_idx} | {mode}"
        else:
            display_mode = "NO_TRACKER"
            label = f"Frame {frame_idx} | NO TRACKER"

        # --- Save visualization frame ---
        if vis_dir:
            vis = frame.copy()
            banner_color = STATUS_COLORS.get(display_mode, (80, 80, 80))

            # Draw segmentation mask overlay (green) if available
            if seg_mask is not None:
                green_overlay = np.zeros_like(vis)
                green_overlay[:, :, 1] = 255
                mask_3ch = np.stack([seg_mask, seg_mask, seg_mask], axis=-1)
                vis = np.where(mask_3ch, cv2.addWeighted(vis, 0.6, green_overlay, 0.4, 0), vis)

            # Draw projected pitch wireframe (red) if available
            if params is not None:
                pp = np.array([frame_w / 2.0, frame_h / 2.0])
                outer_corners = PITCH_CORNERS_3D[:4].copy()
                if yaw_flip:
                    outer_corners[:, 0:2] *= -1
                projected, valid = project_points_batch(params, outer_corners, pp)
                if np.all(valid):
                    corners_2d = projected.astype(np.int32)
                    cv2.polylines(vis, [corners_2d], isClosed=True, color=(0, 0, 255), thickness=3)

            # Draw colored status banner
            cv2.rectangle(vis, (0, 0), (frame_w, 55), banner_color, -1)
            cv2.putText(vis, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(vis, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1)

            vis_path = os.path.join(vis_dir, f"{frame_idx:06d}.jpg")
            cv2.imwrite(vis_path, vis)

        if frames_evaluated % 10 == 0 and frames_evaluated > 0 and frame_idx % 30 == 0:
            print(f"  Evaluated {frames_evaluated} frames, running mean IoU: {np.mean(ious):.3f}")

        if vis_dir and frame_idx % 50 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames...")

    cap.release()

    if not ious:
        print("  No frames could be evaluated (seg model detected no pitch).")
        return None

    ious = np.array(ious)
    mean_iou = np.mean(ious)
    median_iou = np.median(ious)

    print(f"\n  Frames evaluated: {len(ious)}")
    print(f"  Mean IoU:   {mean_iou:.3f}")
    print(f"  Median IoU: {median_iou:.3f}")
    print(f"  Min IoU:    {np.min(ious):.3f}")
    print(f"  Max IoU:    {np.max(ious):.3f}")

    verdict = "PASS" if mean_iou > 0.7 else ("MARGINAL" if mean_iou > 0.4 else "FAIL")
    print(f"\n  Verdict: {verdict}  (PASS > 0.7, MARGINAL > 0.4, FAIL <= 0.4)")
    return mean_iou


# ─────────────────────────────────────────────────────
# TRACKING COVERAGE SUMMARY
# ─────────────────────────────────────────────────────

def run_coverage_summary(frames_data):
    """Summary of tracking modes across the video."""
    print("\n" + "=" * 60)
    print("TRACKING COVERAGE")
    print("=" * 60)

    total = len(frames_data)
    mode_counts = {}
    for key, entry in frames_data.items():
        m = entry.get("mode", "UNKNOWN")
        mode_counts[m] = mode_counts.get(m, 0) + 1

    optimized = mode_counts.get("OPTIMIZED", 0)
    init = mode_counts.get("INIT", 0) + mode_counts.get("REINIT", 0)
    lines_only = mode_counts.get("LINES_ONLY", 0)
    lines_primary = mode_counts.get("LINES_PRIMARY", 0)
    coasting = mode_counts.get("COAST", 0)
    rejected = mode_counts.get("OPT_REJECTED", 0) + mode_counts.get("LINES_REJECTED", 0)
    failed = mode_counts.get("INIT_FAILED", 0)
    none_mode = mode_counts.get("NONE", 0)

    tracked = optimized + init + lines_only + lines_primary
    coverage = tracked / total if total > 0 else 0

    print(f"  Total frames:     {total}")
    print(f"  OPTIMIZED:        {optimized} ({optimized/total*100:.1f}%)")
    print(f"  LINES_PRIMARY:    {lines_primary} ({lines_primary/total*100:.1f}%)")
    print(f"  LINES_ONLY:       {lines_only} ({lines_only/total*100:.1f}%)")
    print(f"  INIT/REINIT:      {init}")
    print(f"  COASTING:         {coasting} ({coasting/total*100:.1f}%)")
    print(f"  REJECTED:         {rejected} ({rejected/total*100:.1f}%)")
    print(f"  INIT_FAILED:      {failed}")
    print(f"  NO TRACKER:       {none_mode}")
    print(f"\n  Tracked frames: {tracked}/{total} ({coverage*100:.1f}%)")

    verdict = "PASS" if coverage > 0.7 else ("MARGINAL" if coverage > 0.4 else "FAIL")
    print(f"  Verdict: {verdict}  (PASS > 70%, MARGINAL > 40%, FAIL <= 40%)")
    return coverage


# ─────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────

def print_final_report(coverage, sanity_rate, mean_reproj, mean_iou, output_path=None):
    """Print overall pass/fail verdict and optionally save to file."""
    report_lines = []
    
    def report_print(s=""):
        print(s)
        report_lines.append(s)

    report_print("\n" + "=" * 60)
    report_print("FINAL REPORT")
    report_print("=" * 60)
    
    results = []

    def grade(name, value, pass_thresh, marginal_thresh, higher_is_better=True):
        if value is None:
            results.append((name, "N/A", "SKIPPED"))
            return
        if higher_is_better:
            v = "PASS" if value > pass_thresh else ("MARGINAL" if value > marginal_thresh else "FAIL")
        else:
            v = "PASS" if value < pass_thresh else ("MARGINAL" if value < marginal_thresh else "FAIL")
        results.append((name, f"{value:.2f}" if isinstance(value, float) else str(value), v))

    grade("Tracking coverage", coverage, 0.7, 0.4, higher_is_better=True)
    grade("Physical sanity rate", sanity_rate, 0.9, 0.7, higher_is_better=True)
    grade("Mean reproj error (px)", mean_reproj, 15, 40, higher_is_better=False)
    grade("Mean Line IoU", mean_iou, 0.7, 0.4, higher_is_better=True)

    for name, val, verdict in results:
        report_print(f"  {name:30s}  {val:>8s}  [{verdict}]")

    verdicts = [v for _, _, v in results if v != "SKIPPED"]
    if not verdicts:
        overall = "SKIPPED — No tests were performed."
    elif all(v == "PASS" for v in verdicts):
        overall = "PASS — Camera parameters are physically reasonable and geometrically consistent."
    elif any(v == "FAIL" for v in verdicts):
        overall = "FAIL — Pipeline is producing unreliable or incorrect camera parameters."
    else:
        overall = "MARGINAL — Some metrics are acceptable but others need improvement."

    report_print(f"\n  OVERALL: {overall}")

    if output_path:
        try:
            with open(output_path, "w") as f:
                f.write("\n".join(report_lines) + "\n")
            print(f"\n  [INFO] Report saved to: {output_path}")
        except Exception as e:
            print(f"\n  [ERROR] Failed to save report to {output_path}: {e}")


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate BroadTrack cricket pipeline output")
    parser.add_argument("-j", "--json", required=True, help="Path to pipeline output JSON")
    parser.add_argument("-v", "--video", help="Path to input video (needed for Line IoU)")
    parser.add_argument("-s", "--seg-model", help="Path to segmentation model (needed for Line IoU)")
    parser.add_argument("--sam-masks", help="Alternative to seg-model: directory containing pre-computed SAM masks")
    parser.add_argument("-o", "--output", help="Path to save the evaluation report (text file)")
    parser.add_argument("--vis-dir", help="Directory to save IoU visualization frames (shows wireframe vs mask overlap)")
    args = parser.parse_args()

    with open(args.json, "r") as f:
        frames_data = json.load(f)

    print(f"\nLoaded {len(frames_data)} frames from {args.json}")

    coverage = run_coverage_summary(frames_data)
    sanity_rate = run_test1_sanity(frames_data)
    mean_reproj = run_test2_reproj(frames_data)
    mean_iou = run_test3_line_iou(
        frames_data, args.video, 
        seg_model_path=args.seg_model, 
        sam_masks_dir=args.sam_masks,
        vis_dir=args.vis_dir
    )
    print_final_report(coverage, sanity_rate, mean_reproj, mean_iou, output_path=args.output)


if __name__ == "__main__":
    main()
