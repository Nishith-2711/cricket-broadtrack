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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cricket.src.camera import project_points_batch, focal_from_hfov
from cricket.src.camera_tracker import PITCH_CORNERS_3D


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

def compute_line_iou(params, seg_mask, image_w, image_h):
    """
    Compute IoU between projected pitch polygon and segmentation mask.
    params: 8-element camera param vector
    seg_mask: binary mask from seg model (H, W), values 0 or 1
    Returns IoU float, or None if projection fails.
    """
    pp = np.array([image_w / 2.0, image_h / 2.0])
    projected, valid = project_points_batch(params, PITCH_CORNERS_3D, pp)

    # Only use the first 4 points (outer pitch corners, indices 0-3).
    # Points 4-7 are popping crease intersections that lie INSIDE the
    # rectangle — including them creates a self-intersecting polygon.
    if not np.all(valid[:4]):
        return None

    corners_2d = projected[:4].astype(np.int32)

    proj_mask = np.zeros((image_h, image_w), dtype=np.uint8)
    cv2.fillPoly(proj_mask, [corners_2d], 1)

    intersection = np.sum(proj_mask & seg_mask)
    union = np.sum(proj_mask | seg_mask)

    if union == 0:
        return None

    return float(intersection) / float(union)


def run_test3_line_iou(frames_data, video_path, seg_model_path):
    """Test 3: Line IoU — projected pitch vs segmentation mask."""
    print("\n" + "=" * 60)
    print("TEST 3: LINE IoU (projected pitch vs seg mask)")
    print("=" * 60)

    if not video_path or not seg_model_path:
        print("  Skipped — requires --video and --seg-model arguments.")
        return None

    try:
        from ultralytics import YOLO
    except ImportError:
        print("  Skipped — ultralytics not installed.")
        return None

    seg_model = YOLO(seg_model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Error: Cannot open video {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sorted_keys = sorted(frames_data.keys(), key=lambda k: frames_data[k].get("time", 0))

    ious = []
    frames_evaluated = 0
    frame_idx = 0

    for key in sorted_keys:
        entry = frames_data[key]
        target_idx = int(entry.get("time", 0))

        # Only evaluate frames with good tracking
        if entry.get("score", 0) < 0.5:
            continue

        cp = entry["cp"]
        if "rvec" not in cp:
            continue

        # Seek to the right frame
        while frame_idx < target_idx:
            cap.read()
            frame_idx += 1

        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Build param vector
        rvec = np.array(cp["rvec"])
        position = np.array([cp["positionXMeters"], cp["positionYMeters"], cp["positionZMeters"]])
        hfov = cp["horizontalFieldOfViewDegrees"]
        focal = focal_from_hfov(hfov, frame_w)

        params = np.zeros(8)
        params[0:3] = rvec
        params[3:6] = position
        params[6] = focal
        params[7] = 0.0

        # Run segmentation
        seg_results = seg_model(frame, conf=0.25, verbose=False)
        sr = seg_results[0]

        if sr.masks is None or len(sr.masks.data) == 0:
            continue

        mask_raw = sr.masks.data[0].cpu().numpy()
        mask_resized = cv2.resize(mask_raw, (frame_w, frame_h))
        seg_mask = (mask_resized > 0.5).astype(np.uint8)

        iou = compute_line_iou(params, seg_mask, frame_w, frame_h)
        if iou is not None:
            ious.append(iou)
            frames_evaluated += 1

        if frames_evaluated % 10 == 0 and frames_evaluated > 0:
            print(f"  Evaluated {frames_evaluated} frames, running mean IoU: {np.mean(ious):.3f}")

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
    coasting = mode_counts.get("COAST", 0)
    rejected = mode_counts.get("OPT_REJECTED", 0)
    failed = mode_counts.get("INIT_FAILED", 0)
    none_mode = mode_counts.get("NONE", 0)

    tracked = optimized + init
    coverage = tracked / total if total > 0 else 0

    print(f"  Total frames:     {total}")
    print(f"  OPTIMIZED:        {optimized} ({optimized/total*100:.1f}%)")
    print(f"  INIT/REINIT:      {init}")
    print(f"  COASTING:         {coasting} ({coasting/total*100:.1f}%)")
    print(f"  OPT_REJECTED:     {rejected} ({rejected/total*100:.1f}%)")
    print(f"  INIT_FAILED:      {failed}")
    print(f"  NO TRACKER:       {none_mode}")
    print(f"\n  Tracked (OPTIMIZED + INIT): {tracked}/{total} ({coverage*100:.1f}%)")

    verdict = "PASS" if coverage > 0.7 else ("MARGINAL" if coverage > 0.4 else "FAIL")
    print(f"  Verdict: {verdict}  (PASS > 70%, MARGINAL > 40%, FAIL <= 40%)")
    return coverage


# ─────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────

def print_final_report(coverage, sanity_rate, mean_reproj, mean_iou):
    """Print overall pass/fail verdict."""
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)

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
        print(f"  {name:30s}  {val:>8s}  [{verdict}]")

    verdicts = [v for _, _, v in results if v != "SKIPPED"]
    if all(v == "PASS" for v in verdicts):
        overall = "PASS — Camera parameters are physically reasonable and geometrically consistent."
    elif any(v == "FAIL" for v in verdicts):
        overall = "FAIL — Pipeline is producing unreliable or incorrect camera parameters."
    else:
        overall = "MARGINAL — Some metrics are acceptable but others need improvement."

    print(f"\n  OVERALL: {overall}")


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate BroadTrack cricket pipeline output")
    parser.add_argument("-j", "--json", required=True, help="Path to pipeline output JSON")
    parser.add_argument("-v", "--video", help="Path to input video (needed for Line IoU)")
    parser.add_argument("-s", "--seg-model", help="Path to segmentation model (needed for Line IoU)")
    args = parser.parse_args()

    with open(args.json, "r") as f:
        frames_data = json.load(f)

    print(f"\nLoaded {len(frames_data)} frames from {args.json}")

    coverage = run_coverage_summary(frames_data)
    sanity_rate = run_test1_sanity(frames_data)
    mean_reproj = run_test2_reproj(frames_data)
    mean_iou = run_test3_line_iou(frames_data, args.video, args.seg_model)
    print_final_report(coverage, sanity_rate, mean_reproj, mean_iou)


if __name__ == "__main__":
    main()
