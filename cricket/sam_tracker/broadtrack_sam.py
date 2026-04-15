"""
BroadTrack-Style Cricket Pitch Camera Tracker — SAM3 Lines-Primary Mode

Based on broadtrack_cricket.py with key changes:
  - Reads pre-computed SAM3 binary masks (--sam-masks)
  - New tracking modes: LINES_PRIMARY (1-2 kpts + edges), LINES_ONLY (0 kpts, edges only)
  - Dynamic line_weight: 1.0 when ≥3 keypoints, 3.0 when 0-2 keypoints
  - Initialization still requires ≥4 YOLO keypoints (PnP)

Usage:
  python sam_tracker/broadtrack_sam.py -v input_videos/input1.mp4 \\
    -m partial_visibility_dataset/best.pt \\
    --sam-masks output_masks/input1_masks \\
    -o output_jsons/input1_sam.json --visualize --vis-dir sam_vis
"""

import cv2
import json
import glob
import argparse
import numpy as np
import os
import sys
import math
from ultralytics import YOLO
from camera_tracker_sam import _clip_segment_to_frame

# Import from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from camera import project_point, project_points_batch, hfov_from_focal, focal_from_hfov

# Import from this directory (sam_tracker)
from camera_tracker_sam import CameraTracker, PITCH_CORNERS_3D, PITCH_WIREFRAME


# ─────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────

def draw_keypoints(img, kpts, kconf, kp_threshold, mode=""):
    """Draw raw YOLO keypoints on frame with confidence values."""
    vis = img.copy()
    labels = ["BatL", "BatR", "BowR", "BowL", "BatLP", "BatRP", "BowRP", "BowLP"]
    
    for i in range(min(len(kpts), 8)):
        x, y = int(kpts[i][0]), int(kpts[i][1])
        conf = kconf[i] if i < len(kconf) else 0.0
        
        if x == 0 and y == 0:
            continue
        
        if conf > kp_threshold:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        
        cv2.circle(vis, (x, y), 8, color, -1)
        cv2.circle(vis, (x, y), 8, (255, 255, 255), 2)
        label = f"{labels[i]} {conf:.2f}"
        cv2.putText(vis, label, (x + 12, y - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return vis


# ─────────────────────────────────────────────────────
# HISTOGRAM CAMERA CUT DETECTION
# ─────────────────────────────────────────────────────

def detect_camera_cut(prev_gray, curr_gray, threshold=0.4):
    """Detect camera cut via histogram correlation + mean absolute difference.

    Histogram alone fails on cricket footage because both sides of a cut share
    similar outdoor green-grass brightness distributions.  Adding a pixel-level
    MAD check catches hard cuts that histogram misses.
    """
    hist1 = cv2.calcHist([prev_gray], [0], None, [64], [0, 256])
    hist2 = cv2.calcHist([curr_gray], [0], None, [64], [0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    # Mean absolute pixel difference (normalised 0-1).
    # A hard broadcast cut typically produces MAD > 0.15.
    mad = np.mean(np.abs(prev_gray.astype(np.float32) - curr_gray.astype(np.float32))) / 255.0

    return hist_score < threshold or mad > 0.15


# ─────────────────────────────────────────────────────
# LINE EXTRACTION (FROM SEGMENTATION MASK)
# ─────────────────────────────────────────────────────

def extract_pitch_edges_from_mask(mask, num_points=80, border_margin=50):
    """
    Given a binary mask of the pitch, extracts sample points along its logical boundaries.
    Uses convex hull and polygon approximation to ignore player overlap intrusions.
    
    CRITICAL: Points near the frame border are FILTERED OUT AGGRESSIVELY. When the pitch
    is only partially visible (e.g., zoomed in), the mask edge where it exits the frame
    is a crop artifact, not a real pitch boundary. Including those points causes the
    optimizer to think a clipped edge is real, leading to position/zoom errors.
    
    IMPROVEMENT: Increased default border_margin to 50px (was 30px) and added geometric
    filtering — edges aligned with frame edges are considered artifacts.
    """
    if mask is None or np.sum(mask) == 0:
        return []

    img_h, img_w = mask.shape[:2]
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    largest_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest_contour)
    epsilon = 0.005 * cv2.arcLength(hull, True)
    approx_poly = cv2.approxPolyDP(hull, epsilon, True)
    
    poly_vertices = approx_poly.reshape(-1, 2)
    n_vertices = len(poly_vertices)
    
    if n_vertices < 3:
        return []
        
    sampled_points = []
    points_per_edge = max(1, num_points // n_vertices)
    
    for i in range(n_vertices):
        p1 = poly_vertices[i].astype(float)
        p2 = poly_vertices[(i + 1) % n_vertices].astype(float)
        
        # GEOMETRIC FILTER: Skip edges that are aligned with frame edges (artifacts)
        dx = abs(p2[0] - p1[0])
        dy = abs(p2[1] - p1[1])
        is_horizontal = (dy < 5)
        is_vertical = (dx < 5)
        
        # If edge is mostly at frame boundary (e.g., top=0, bottom=h-1), skip entire edge
        at_frame_top = (p1[1] < 5 and p2[1] < 5)
        at_frame_bottom = (p1[1] > img_h - 5 and p2[1] > img_h - 5)
        at_frame_left = (p1[0] < 5 and p2[0] < 5)
        at_frame_right = (p1[0] > img_w - 5 and p2[0] > img_w - 5)
        
        if at_frame_top or at_frame_bottom or at_frame_left or at_frame_right:
            continue  # Entire edge is a frame border artifact
        
        for j in range(points_per_edge):
            alpha = j / points_per_edge
            pt = p1 * (1.0 - alpha) + p2 * alpha
            x, y = int(pt[0]), int(pt[1])
            
            # Aggressive interior-only filtering (default margin doubled)
            if x < border_margin or x > (img_w - border_margin):
                continue
            if y < border_margin or y > (img_h - border_margin):
                continue
            
            sampled_points.append((x, y))
    
    return sampled_points


def mask_touches_border(mask, border_margin=8):
    """Return True when the segmentation mask touches the frame border.

    This is a strong indicator that the pitch is partially visible and the
    tracker should prefer a line-dominant / partial-visibility regime.
    """
    if mask is None or np.sum(mask) == 0:
        return False

    h, w = mask.shape[:2]
    bm = max(1, int(border_margin))
    return (
        np.any(mask[:bm, :] > 0) or
        np.any(mask[h - bm:, :] > 0) or
        np.any(mask[:, :bm] > 0) or
        np.any(mask[:, w - bm:] > 0)
    )


# ─────────────────────────────────────────────────────
# OPTICAL FLOW HELPERS
# ─────────────────────────────────────────────────────

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
MIN_OF_FEATURES = 8


def backproject_to_ground(pts_2d, params, principal_point):
    """Back-project 2D pixel coordinates to 3D points on the Z=0 ground plane."""
    rvec = params[0:3]
    position = params[3:6]
    focal = params[6]

    R, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float64))
    K = np.array([
        [focal, 0, principal_point[0]],
        [0, focal, principal_point[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    K_inv = np.linalg.inv(K)

    world_pts = []
    for pt in pts_2d:
        uv1 = np.array([pt[0], pt[1], 1.0])
        ray_cam = K_inv @ uv1
        ray_world = R.T @ ray_cam
        if abs(ray_world[2]) > 1e-6:
            s = -position[2] / ray_world[2]
            intersection = position + s * ray_world
            world_pts.append([intersection[0], intersection[1], 0.0])
        else:
            world_pts.append([0.0, 0.0, 0.0])

    return np.array(world_pts, dtype=np.float64)


def extract_of_features(gray, params, principal_point, image_w, image_h, yaw_flip=False):
    """Extract Shi-Tomasi features within the projected pitch polygon."""
    outer_corners = PITCH_CORNERS_3D[:4].copy()
    if yaw_flip:
        outer_corners[:, 0:2] *= -1

    corners_2d, valid = project_points_batch(params, outer_corners, principal_point)
    if not np.all(valid):
        return None, None

    mask = np.zeros((image_h, image_w), dtype=np.uint8)
    poly = corners_2d.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [poly], 255)

    features = cv2.goodFeaturesToTrack(
        gray, mask=mask, maxCorners=100, qualityLevel=0.01, minDistance=10
    )
    if features is None or len(features) < MIN_OF_FEATURES:
        return None, None

    pts_2d = features.reshape(-1, 2)
    pts_3d = backproject_to_ground(pts_2d, params, principal_point)

    return features, pts_3d


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BroadTrack SAM3 Lines-Primary Cricket Tracker")
    parser.add_argument('-v', '--video', help='Path to input video')
    parser.add_argument('-i', '--images', help='Path to input image folder')
    parser.add_argument('-m', '--model', required=True, help='Path to YOLO pose model')
    parser.add_argument('--sam-masks', required=True, help='Path to directory of SAM3 binary mask PNGs (from sam_detector.py --save-masks)')
    parser.add_argument('-o', '--output', required=True, help='Path to output JSON')
    parser.add_argument('--conf', type=float, default=0.25, help='YOLO box confidence threshold')
    parser.add_argument('--kp-conf', type=float, default=0.7, help='Per-keypoint confidence threshold')
    parser.add_argument('--use-predictive-anchors', action='store_true', help='Hallucinate missing keypoints from previous camera matrix')
    parser.add_argument('--ema-alpha', type=float, default=0.2, help='EMA smoothing factor (0.0 to 1.0, lower is smoother)')
    parser.add_argument('--reinit-after', type=int, default=30, help='Force re-init after N consecutive failures')
    parser.add_argument('--use-optical-flow', action='store_true', help='Use Lucas-Kanade optical flow to bridge gaps')
    parser.add_argument('--tripod', help='Path to tripod.json (enables Pass 2 locked-position mode)')
    parser.add_argument('--rotation-penalty', type=float, default=10.0, help='Rotation change penalty weight')
    parser.add_argument('--visualize', action='store_true', help='Save visualization frames')
    parser.add_argument('--vis-dir', default='sam_vis', help='Visualization output dir')
    parser.add_argument('--use-lines', action='store_true', default=True, help='Use SAM edge constraints (default: True)')
    parser.add_argument('--no-lines', dest='use_lines', action='store_false', help='Disable SAM edge constraints for debugging')
    args = parser.parse_args()

    if not args.video and not args.images:
        print("Error: Must provide either --video or --images")
        return

    # Load YOLO model (keypoints only — no seg model needed)
    model = YOLO(args.model)
    model.to("cpu")
    # Load SAM3 pre-computed masks
    sam_mask_files = sorted(glob.glob(os.path.join(args.sam_masks, 'mask_*.png')))
    if len(sam_mask_files) == 0:
        print(f"ERROR: No mask_*.png files found in {args.sam_masks}")
        print(f"  Run sam_detector.py with --save-masks first.")
        return
    print(f"  SAM3 MASKS: loaded {len(sam_mask_files)} pre-computed masks from {args.sam_masks}")

    # Load tripod parameters (Pass 2 mode)
    tripod_position = None
    tripod_radius = None
    if args.tripod:
        with open(args.tripod, 'r') as f:
            tripod_data = json.load(f)
        tripod_position = np.array(tripod_data['sphere']['center'])
        tripod_radius = float(tripod_data['sphere']['radius'])
        print(f"  TRIPOD LOCK: center=[{tripod_position[0]:.2f}, {tripod_position[1]:.2f}, {tripod_position[2]:.2f}], radius={tripod_radius:.4f}m")

    is_video = False
    if args.video:
        is_video = True
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"Error: Cannot open video {args.video}")
            return
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        exts = ['.jpg', '.jpeg', '.png', '.bmp']
        frames_source = sorted([f for f in os.listdir(args.images)
                                if os.path.splitext(f)[1].lower() in exts])
        total_frames = len(frames_source)

    if args.visualize:
        os.makedirs(args.vis_dir, exist_ok=True)

    # State
    tracker = None
    prev_gray = None
    frames_data = {}
    count = 0
    consecutive_failures = 0
    
    # Persistent state for temporally smoothing the raw YOLO keypoints
    ema_kpts = [None] * 8
    
    # Persistent state for temporally smoothing SAM masks
    ema_mask = None

    # Optical flow state
    of_pts = None
    of_3d = None

    print(f"\n{'='*60}")
    print("BROADTRACK SAM3 LINES-PRIMARY CRICKET TRACKER")
    print(f"{'='*60}")

    while True:
        # 1. Read frame
        if is_video:
            ret, frame = cap.read()
            if not ret:
                break
            filename = f"frame_{count:06d}.jpg"
        else:
            if count >= total_frames:
                break
            filename = frames_source[count]
            frame = cv2.imread(os.path.abspath(os.path.join(args.images, filename)))
            if frame is None:
                count += 1
                continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]
        key = f"/workspace/cricket/{filename}"

        # Base frame entry
        frame_entry = {
            "cp": {
                "sensorResolutionWidthPixels": float(w),
                "sensorResolutionHeightPixels": float(h),
                "horizontalFieldOfViewDegrees": 30.0,
                "normalizedRadialDistortionCoefficients": [0.0]
            },
            "score": 0.0,
            "time": float(count)
        }

        mode = "NONE"
        reproj_err = None

        # 2. Camera cut detection
        camera_cut = False
        if prev_gray is not None:
            camera_cut = detect_camera_cut(prev_gray, gray)
            if camera_cut:
                print(f"  [Frame {count}] CAMERA CUT — resetting tracker")
                tracker = None
                consecutive_failures = 0
                of_pts = None
                of_3d = None
                ema_mask = None       # Reset SAM mask accumulator
                ema_kpts = [None] * 8 # Reset keypoint EMA too

        # 3. Run YOLO (keypoints only)
        results = model(frame, conf=args.conf, verbose=False, device="cpu")
        r = results[0]

        # ─── SAM3 Mask Loading ───
        edge_points = []
        partial_visibility = False
        if count < len(sam_mask_files):
            mask_raw = cv2.imread(sam_mask_files[count], cv2.IMREAD_GRAYSCALE)
            if mask_raw is not None:
                # Resize to match frame if needed
                if mask_raw.shape[0] != h or mask_raw.shape[1] != w:
                    mask_raw = cv2.resize(mask_raw, (w, h))
                mask_resized = mask_raw.astype(np.float32) / 255.0

                # Temporally smooth
                if ema_mask is None:
                    ema_mask = mask_resized.copy()
                else:
                    ema_mask = args.ema_alpha * mask_resized + (1 - args.ema_alpha) * ema_mask

        if ema_mask is not None:
            mask_binary = (ema_mask > 0.5).astype(np.uint8) * 255
            partial_visibility = mask_touches_border(mask_binary, border_margin=8)
            edge_points = extract_pitch_edges_from_mask(mask_binary, num_points=80)

        # ─── YOLO Keypoint Extraction ───
        keypoints_2d = None
        keypoints_3d = None
        n_valid = 0

        if r.keypoints is not None and len(r.keypoints.xy) > 0 and r.keypoints.xy.shape[1] >= 8:
            kpts = r.keypoints.xy[0].cpu().numpy()
            kconf = r.keypoints.conf[0].cpu().numpy() if r.keypoints.conf is not None else np.ones(8)

            # Apply Temporal Smoothing (EMA Filter)
            for i in range(8):
                if i < len(kconf) and kconf[i] > 0.05:
                    if ema_kpts[i] is None:
                        ema_kpts[i] = kpts[i].copy()
                    else:
                        ema_kpts[i] = args.ema_alpha * kpts[i] + (1 - args.ema_alpha) * ema_kpts[i]
                    kpts[i] = ema_kpts[i]

            kp_threshold = args.kp_conf
            valid_mask = kconf > kp_threshold
            n_valid = int(np.sum(valid_mask))

            if n_valid >= 1:  # Accept even 1 keypoint (lines will supplement)
                keypoints_2d = kpts[valid_mask][:n_valid].astype(np.float64)
                keypoints_3d = PITCH_CORNERS_3D[valid_mask][:n_valid]

        # ─── 4. Initialize or update tracker ───
        has_edges = len(edge_points) >= 6
        visible_corners = 0

        if tracker is None or not tracker.initialized:
            # Initialization still requires ≥4 keypoints for PnP
            if keypoints_2d is not None and n_valid >= 4:
                tracker = CameraTracker(w, h, tripod_position=tripod_position, tripod_radius=tripod_radius)
                success = tracker.reinit(keypoints_2d, keypoints_3d)
                if success:
                    mode = "INIT"
                    consecutive_failures = 0
                    info = tracker.get_camera_info()
                    print(f"  [Frame {count}] INITIALIZED: Z={info['position'][2]:.1f}m, "
                          f"hfov={info['hfov_deg']:.0f}°")
                    
                    if args.use_optical_flow and tracker.params is not None:
                        of_pts, of_3d = extract_of_features(
                            gray, tracker.params, tracker.principal_point, w, h,
                            yaw_flip=tracker.yaw_flip
                        )
                else:
                    tracker = None
                    mode = "INIT_FAILED"
                    if count % 10 == 0:
                        print(f"  [Frame {count}] INIT FAILED ({n_valid} keypoints)")
        else:
            # ─── CHANGE 4: Lines-primary tracking modes ───
            
            # IMPROVED: Check if corners are visible (handles zoom-induced partial visibility)
            if tracker is not None and tracker.params is not None:
                from camera_tracker_sam import get_visible_corner_count
                pitch_corners = PITCH_CORNERS_3D
                if tracker.yaw_flip:
                    pitch_corners = pitch_corners.copy()
                    pitch_corners[:, 0:2] *= -1
                
                visible_corners, all_visible = get_visible_corner_count(
                    tracker.params, pitch_corners, tracker.principal_point, w, h
                )
            else:
                visible_corners = 0
                all_visible = False
            
            # Determine tracking signal availability
            yolo_succeeded = keypoints_2d is not None and (n_valid >= 3 or (args.use_predictive_anchors and n_valid > 0))
            lines_can_track = has_edges  # SAM mask provides edge constraints
            
            # Compute dynamic weights based on visibility + available signal.
            #
            # With Option B in place, line residuals fit the dirt-strip
            # rectangle (which matches what SAM actually segments), so lines
            # are geometrically correct — but there are still ~50-100 edge
            # points vs. 8 keypoints, so raw residual counts would swamp the
            # keypoint signal. These weights are chosen so the keypoints
            # dominate when they exist, and lines take over only when the
            # keypoint count collapses.
            # Compute zoom penalty based on visible corners (used for dynamic weighting)
            corner_visibility_factor = visible_corners / 8.0  # 0-1: how many corners are visible
            zoom_penalty = 1.0 - corner_visibility_factor  # 0 if all visible, 1 if none visible
            if partial_visibility:
                zoom_penalty = max(zoom_penalty, 0.75)

            if has_edges:

                if n_valid <= 1:
                    # Very few keypoints: lines are PRIMARY
                    effective_line_weight = 3.0 + (1.5 * zoom_penalty)
                    effective_kp_weight = 0.3
                elif n_valid <= 2:
                    # Few keypoints: lines lead but keypoints still matter
                    effective_line_weight = 1.8 + (1.0 * zoom_penalty)
                    effective_kp_weight = 0.6
                else:
                    # ≥3 keypoints: TRUST the keypoints, lines just nudge.
                    # Previous setting (1.5 + zoom_penalty) was drowning the
                    # ~8 keypoints under ~80 edge residuals and causing
                    # 44–60 px drift in OPTIMIZED mode.
                    effective_line_weight = 0.35 + (0.35 * zoom_penalty)
                    effective_kp_weight = 1.0
            else:
                effective_line_weight = 0.2 + (2.0 * zoom_penalty)
                effective_kp_weight = 1.0

            if yolo_succeeded and not partial_visibility and (all_visible or visible_corners >= 4):
                # OPTIMIZED mode: ≥3 keypoints + most corners visible
                # IMPROVED: Adaptive outlier threshold based on visibility
                adaptive_outlier_threshold = 30.0 + (15.0 * (1.0 - visible_corners / 8.0))
                
                reproj_err, new_params, n_kp_used = tracker.update(
                    keypoints_2d, keypoints_3d,
                    rotation_weight=args.rotation_penalty,
                    outlier_threshold=adaptive_outlier_threshold,
                    edge_points=edge_points,
                    use_lines=True,
                    use_predictive_anchors=args.use_predictive_anchors,
                    line_weight=effective_line_weight,
                    kp_weight_scale=effective_kp_weight,
                    partial_visibility=partial_visibility
                )

                if reproj_err is not None and n_kp_used >= 3:
                    mode = "OPTIMIZED"
                    consecutive_failures = 0
                    if count % 10 == 0:
                        info = tracker.get_camera_info()
                        print(f"  [Frame {count}] OPTIMIZED: reproj={reproj_err:.1f}px, vis_corners={visible_corners}/8, "
                              f"Z={info['position'][2]:.1f}m, hfov={info['hfov_deg']:.0f}°")

                    if args.use_optical_flow and tracker.params is not None:
                        of_pts, of_3d = extract_of_features(
                            gray, tracker.params, tracker.principal_point, w, h,
                            yaw_flip=tracker.yaw_flip
                        )
                        if of_pts is not None and count % 30 == 0:
                            print(f"  [Frame {count}] OF refreshed: {len(of_pts)} features")
                elif reproj_err is not None and n_kp_used < 3:
                    # Predictive validation dropped most keypoints — the fit
                    # used mostly lines.  Downgrade to LINES_PRIMARY so we
                    # don't falsely label a line-dominated fit as OPTIMIZED.
                    mode = "LINES_PRIMARY"
                    consecutive_failures = 0
                    if count % 10 == 0:
                        info = tracker.get_camera_info()
                        print(f"  [Frame {count}] LINES_PRIMARY (downgraded): reproj={reproj_err:.1f}px, "
                              f"{n_kp_used} kpts survived, vis_corners={visible_corners}/8, "
                              f"Z={info['position'][2]:.1f}m, hfov={info['hfov_deg']:.0f}°")
                else:
                    mode = "OPT_REJECTED"
                    consecutive_failures += 1
                    if count % 10 == 0:
                        print(f"  [Frame {count}] OPT REJECTED (vis={visible_corners}/8)")

            elif lines_can_track:
                # ─── LINES_PRIMARY / LINES_ONLY mode ───
                # Feed whatever keypoints we have (0-2) + edge constraints
                kp_2d_feed = keypoints_2d if keypoints_2d is not None else np.empty((0, 2), dtype=np.float64)
                kp_3d_feed = keypoints_3d if keypoints_3d is not None else np.empty((0, 3), dtype=np.float64)
                
                reproj_err, new_params, n_kp_used = tracker.update(
                    kp_2d_feed, kp_3d_feed,
                    rotation_weight=args.rotation_penalty,
                    edge_points=edge_points,
                    use_lines=True,
                    use_predictive_anchors=False,  # Don't hallucinate in lines mode
                    line_weight=effective_line_weight,
                    kp_weight_scale=effective_kp_weight,
                    partial_visibility=partial_visibility
                )

                if reproj_err is not None:
                    mode = f"LINES_{'PRIMARY' if n_valid > 0 else 'ONLY'}"
                    consecutive_failures = 0
                    if count % 10 == 0:
                        info = tracker.get_camera_info()
                        print(f"  [Frame {count}] {mode}: cost={reproj_err:.1f}, "
                              f"{n_valid} kpts, {len(edge_points)} edges, vis_corners={visible_corners}/8, "
                              f"Z={info['position'][2]:.1f}m, hfov={info['hfov_deg']:.0f}°")
                else:
                    mode = "LINES_REJECTED"
                    consecutive_failures += 1
                    if count % 10 == 0:
                        print(f"  [Frame {count}] LINES_REJECTED: {len(edge_points)} edge points")

            elif args.use_optical_flow and of_pts is not None and prev_gray is not None and tracker.params is not None:
                # YOLO failed, no edges — try optical flow bridging
                new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, of_pts, None, **LK_PARAMS
                )
                good = status.flatten() == 1
                if np.sum(good) >= MIN_OF_FEATURES:
                    of_pts = new_pts[good].reshape(-1, 1, 2)
                    of_3d = of_3d[good]

                    flow_2d = of_pts.reshape(-1, 2).astype(np.float64)
                    flow_3d = of_3d.astype(np.float64)

                    if tracker.yaw_flip:
                        flow_3d = flow_3d.copy()
                        flow_3d[:, 0] *= -1
                        flow_3d[:, 1] *= -1

                    reproj_err, new_params, n_kp_used = tracker.update(
                        flow_2d, flow_3d,
                        rotation_weight=5.0,
                        edge_points=edge_points,
                        use_lines=True if has_edges else False,
                        line_weight=effective_line_weight if has_edges else 1.0,
                        partial_visibility=partial_visibility
                    )

                    if reproj_err is not None:
                        mode = "FLOW"
                        consecutive_failures = 0
                        if count % 10 == 0:
                            info = tracker.get_camera_info()
                            print(f"  [Frame {count}] FLOW: reproj={reproj_err:.1f}px, "
                                  f"{len(of_pts)} pts, Z={info['position'][2]:.1f}m")
                    else:
                        mode = "FLOW_REJECTED"
                        consecutive_failures += 1
                        of_pts = None
                        of_3d = None
                else:
                    mode = "FLOW_LOST"
                    consecutive_failures += 1
                    of_pts = None
                    of_3d = None

            else:
                mode = "COAST"
                consecutive_failures += 1
                if count % 30 == 0:
                    print(f"  [Frame {count}] COASTING ({n_valid} keypoints, {len(edge_points)} edges)")

            # Re-initialize if failing for too long
            if consecutive_failures >= args.reinit_after and n_valid >= 4:
                tracker = CameraTracker(w, h, tripod_position=tripod_position, tripod_radius=tripod_radius)
                success = tracker.reinit(keypoints_2d, keypoints_3d)
                if success:
                    mode = "REINIT"
                    consecutive_failures = 0
                    ema_kpts = [None] * 8
                    of_pts = None
                    of_3d = None
                    info = tracker.get_camera_info()
                    print(f"  [Frame {count}] RE-INITIALIZED: Z={info['position'][2]:.1f}m, "
                          f"hfov={info['hfov_deg']:.0f}°")
                else:
                    tracker = None

        # 5. Store results
        frame_entry["mode"] = mode
        frame_entry["n_keypoints"] = n_valid
        frame_entry["n_edges"] = len(edge_points)
        frame_entry["partial_visibility"] = bool(partial_visibility)
        frame_entry["visible_corners"] = int(visible_corners) if tracker is not None and tracker.initialized else 0

        if tracker is not None and tracker.initialized and tracker.params is not None:
            params = tracker.params
            info = tracker.get_camera_info()
            
            frame_entry["cp"].update({
                "positionXMeters": info["position"][0],
                "positionYMeters": info["position"][1],
                "positionZMeters": info["position"][2],
                "horizontalFieldOfViewDegrees": info["hfov_deg"],
                "rvec": params[0:3].tolist(),
                "tvec": params[3:6].tolist(),
                "yaw_flip": tracker.yaw_flip,
            })

            if mode == "INIT" or mode == "REINIT":
                frame_entry["score"] = 1.0
            elif mode == "OPTIMIZED":
                frame_entry["score"] = 0.9
                frame_entry["reproj_error_px"] = float(reproj_err) if reproj_err is not None else None
            elif mode.startswith("LINES_"):
                frame_entry["score"] = 0.45  # Exclude from tripod fitting by default (min-score=0.6)
                frame_entry["line_cost"] = float(reproj_err) if reproj_err is not None else None
            elif mode == "FLOW":
                frame_entry["score"] = 0.8
                frame_entry["reproj_error_px"] = float(reproj_err) if reproj_err is not None else None
            elif mode == "COAST":
                frame_entry["score"] = 0.1
            elif mode.endswith("REJECTED"):
                frame_entry["score"] = 0.0

        # 6. Visualization
        if args.visualize:
            if r.keypoints is not None and len(r.keypoints.xy) > 0 and r.keypoints.xy.shape[1] >= 8:
                all_kpts = r.keypoints.xy[0].cpu().numpy()
                all_kconf = r.keypoints.conf[0].cpu().numpy() if r.keypoints.conf is not None else np.ones(8)
                vis = draw_keypoints(frame, all_kpts, all_kconf, args.kp_conf, mode)
            else:
                vis = frame.copy()

            # Draw Pitch Edges (red polygon + yellow dots)
            if len(edge_points) > 1:
                pts = np.array(edge_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
            for (px, py) in edge_points:
                cv2.circle(vis, (int(px), int(py)), 4, (0, 255, 255), -1)

            # Draw projected wireframe if tracker is active
            if tracker is not None and tracker.initialized:
                # Dirt strip (larger rectangle used for line residuals) in CYAN.
                # This is what the line cost is fitting to the SAM mask — it
                # should sit on the red-dashed SAM outline, not inside it.
                dirt_wf_2d = tracker.get_dirt_strip_2d()
                for start_2d, end_2d in dirt_wf_2d:
                    clipped = _clip_segment_to_frame(start_2d, end_2d, w, h)
                    if clipped is None:
                        continue
                    p1, p2 = clipped
                    cv2.line(vis, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 255, 0), 2)

                # Official playing pitch (used for keypoint residuals) in MAGENTA.
                wireframe_2d = tracker.get_wireframe_2d()
                for start_2d, end_2d in wireframe_2d:
                    clipped = _clip_segment_to_frame(start_2d, end_2d, w, h)

                    if clipped is None:
                        continue

                    p1, p2 = clipped
                    cv2.line(vis, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 255), 2)
                
            # Mode + edge count in header
            mode_color = (0, 255, 0) if mode.startswith("LINES") else (0, 255, 255)
            cv2.putText(vis, f"Frame: {count} | {mode} | {n_valid}kp {len(edge_points)}edges", (30, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
            cv2.imwrite(os.path.join(args.vis_dir, f"{count:06d}.jpg"), vis)

        frames_data[key] = frame_entry
        prev_gray = gray.copy()
        count += 1

        if count % 100 == 0:
            print(f"  Processed {count}/{total_frames} frames...")

    if is_video:
        cap.release()

    # Save
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(frames_data, f, indent=4)

    # Print mode summary
    modes = [v.get("mode", "?") for v in frames_data.values()]
    mode_counts = {m: modes.count(m) for m in sorted(set(modes))}
    
    print(f"\n{'='*60}")
    print(f"Saved {len(frames_data)} frames to {args.output}")
    print(f"Mode summary: {mode_counts}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
