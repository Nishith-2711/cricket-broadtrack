"""
BroadTrack-Style Cricket Pitch Camera Tracker

Main pipeline:
  1. YOLO detects 4 corners → keypoint observations
  2. CameraTracker optimizes camera params to minimize reprojection error
  3. Camera position is soft-constrained (tripod model)
  4. Projects full pitch wireframe for visualization

Usage:
  python broadtrack_cricket.py -v input_videos/input1.mp4 \
    -m partial_visibility_dataset/best.pt \
    -o bt_output.json --visualize --vis-dir bt_vis
"""

import cv2
import json
import argparse
import numpy as np
import os
import sys
import math
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from camera import project_point, project_points_batch, hfov_from_focal, focal_from_hfov
from camera_tracker import CameraTracker, PITCH_CORNERS_3D, PITCH_WIREFRAME


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
        
        # Skip if point is at origin (no detection)
        if x == 0 and y == 0:
            continue
        
        # Green = above threshold (used), Red = below threshold (rejected)
        if conf > kp_threshold:
            color = (0, 255, 0)  # Green — accepted
        else:
            color = (0, 0, 255)  # Red — rejected
        
        cv2.circle(vis, (x, y), 8, color, -1)
        cv2.circle(vis, (x, y), 8, (255, 255, 255), 2)  # White border
        label = f"{labels[i]} {conf:.2f}"
        cv2.putText(vis, label, (x + 12, y - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return vis



# ─────────────────────────────────────────────────────
# HISTOGRAM CAMERA CUT DETECTION
# ─────────────────────────────────────────────────────

def detect_camera_cut(prev_gray, curr_gray, threshold=0.4):
    """Detect camera cut via histogram correlation."""
    hist1 = cv2.calcHist([prev_gray], [0], None, [64], [0, 256])
    hist2 = cv2.calcHist([curr_gray], [0], None, [64], [0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return score < threshold


# ─────────────────────────────────────────────────────
# LINE EXTRACTION (FROM SEGMENTATION MASK)
# ─────────────────────────────────────────────────────

def extract_pitch_edges_from_mask(mask, num_points=20):
    """
    Given a binary mask of the pitch, extracts sample points along its logical boundaries.
    Uses convex hull and polygon approximation to ignore player overlap intrusions.
    mask: 2D numpy array (H, W) with values 0 or 255.
    Returns: list of (x, y) coordinates of boundary points.
    """
    if mask is None or np.sum(mask) == 0:
        return []

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # Get the largest contour (assuming it's the pitch)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 1. Convex Hull: Snaps a rubber band around the contour, eliminating "bites" from players standing on the edge.
    hull = cv2.convexHull(largest_contour)
    
    # 2. Polygon Approximation: Forces it into a simpler geometric shape (ideally 4-sided quadrilateral)
    # Using a much tighter epsilon (0.005 instead of 0.02) to prevent the lines from "cutting corners" and intruding into the pitch
    epsilon = 0.005 * cv2.arcLength(hull, True)
    approx_poly = cv2.approxPolyDP(hull, epsilon, True)
    
    # The polygon vertices
    poly_vertices = approx_poly.reshape(-1, 2)
    n_vertices = len(poly_vertices)
    
    if n_vertices < 3:
        return []
        
    # 3. Interpolate points evenly along the perimeter of the simplified polygon
    sampled_points = []
    points_per_edge = max(1, num_points // n_vertices)
    
    for i in range(n_vertices):
        p1 = poly_vertices[i].astype(float)
        p2 = poly_vertices[(i + 1) % n_vertices].astype(float)
        
        for j in range(points_per_edge):
            alpha = j / points_per_edge
            pt = p1 * (1.0 - alpha) + p2 * alpha
            sampled_points.append((int(pt[0]), int(pt[1])))
    
    return sampled_points


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
    """
    Back-project 2D pixel coordinates to 3D points on the Z=0 ground plane
    using the current camera parameters.

    Args:
        pts_2d: (N, 2) array of pixel coordinates
        params: 8-element camera param vector [rvec(3), pos(3), focal, k1]
        principal_point: (2,) array [cx, cy]

    Returns:
        (N, 3) float64 array of world coordinates with Z=0
    """
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
        # Intersection with Z=0 plane: position[2] + s * ray_world[2] = 0
        if abs(ray_world[2]) > 1e-6:
            s = -position[2] / ray_world[2]
            intersection = position + s * ray_world
            world_pts.append([intersection[0], intersection[1], 0.0])
        else:
            world_pts.append([0.0, 0.0, 0.0])

    return np.array(world_pts, dtype=np.float64)


def extract_of_features(gray, params, principal_point, image_w, image_h, yaw_flip=False):
    """
    Extract Shi-Tomasi features within the projected pitch polygon.
    Returns (tracked_pts, tracked_3d) or (None, None) if projection fails.
    """
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
    parser = argparse.ArgumentParser(description="BroadTrack-style cricket camera tracker")
    parser.add_argument('-v', '--video', help='Path to input video')
    parser.add_argument('-i', '--images', help='Path to input image folder')
    parser.add_argument('-m', '--model', required=True, help='Path to YOLO pose model')
    parser.add_argument('-s', '--seg-model', help='Path to YOLO segmentation model')
    parser.add_argument('-o', '--output', required=True, help='Path to output JSON')
    parser.add_argument('--conf', type=float, default=0.25, help='YOLO box confidence threshold')
    parser.add_argument('--kp-conf', type=float, default=0.7, help='Per-keypoint confidence threshold')
    parser.add_argument('--use-lines', action='store_true', help='Use segmentation mask lines for Soft-Tripod constraints')
    parser.add_argument('--use-predictive-anchors', action='store_true', help='Hallucinate missing keypoints from previous camera matrix')
    parser.add_argument('--ema-alpha', type=float, default=0.2, help='EMA smoothing factor for YOLO keypoints (0.0 to 1.0, lower is smoother)')
    parser.add_argument('--reinit-after', type=int, default=30, help='Force re-initialization after this many consecutive failed frames')
    parser.add_argument('--use-optical-flow', action='store_true', help='Use Lucas-Kanade optical flow to bridge keypoint gaps')
    parser.add_argument('--tripod', help='Path to tripod.json from compute_tripod.py (enables Pass 2 locked-position mode)')
    parser.add_argument('--visualize', action='store_true', help='Save visualization frames')
    parser.add_argument('--vis-dir', default='bt_vis', help='Visualization output dir')
    args = parser.parse_args()

    if not args.video and not args.images:
        print("Error: Must provide either --video or --images")
        return

    # Load
    model = YOLO(args.model)
    seg_model = YOLO(args.seg_model) if args.seg_model else None

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
    
    # Persistent state for temporally smoothing YOLO segmentation masks (Method A Lines)
    ema_mask = None

    # Optical flow state
    of_pts = None    # (N, 1, 2) float32 — 2D tracked feature positions
    of_3d = None     # (N, 3) float64 — corresponding 3D world coords on Z=0

    print(f"\n{'='*60}")
    print("BROADTRACK-STYLE CRICKET CAMERA TRACKER")
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

        # 3. Run YOLO
        results = model(frame, conf=args.conf, verbose=False)
        r = results[0]

        # Segmentation Inference
        edge_points = []
        if seg_model is not None:
            seg_results = seg_model(frame, conf=0.25, verbose=False)
            sr = seg_results[0]
            
            mask_resized = None
            if sr.masks is not None and len(sr.masks.data) > 0:
                # Resize the probability map to full HD
                mask_raw = sr.masks.data[0].cpu().numpy()
                mask_resized = cv2.resize(mask_raw, (w, h))
                
            # Temporally Smooth the Mask to drastically reduce boundary flickering in Method A
            if mask_resized is not None:
                if ema_mask is None:
                    ema_mask = mask_resized.copy()
                else:
                    ema_mask = args.ema_alpha * mask_resized + (1 - args.ema_alpha) * ema_mask
            
            # If the tracker has an active smoothed mask, draw the polygon lines
            if ema_mask is not None:
                mask_binary = (ema_mask > 0.5).astype(np.uint8) * 255
                edge_points = extract_pitch_edges_from_mask(mask_binary, num_points=20)

        keypoints_2d = None
        keypoints_3d = None
        n_valid = 0

        if r.keypoints is not None and len(r.keypoints.xy) > 0 and r.keypoints.xy.shape[1] >= 8:
            kpts = r.keypoints.xy[0].cpu().numpy()
            kconf = r.keypoints.conf[0].cpu().numpy() if r.keypoints.conf is not None else np.ones(8)

            # Apply Temporal Smoothing (EMA Filter) to stabilize the raw YOLO detections
            for i in range(8):
                if i < len(kconf) and kconf[i] > 0.05:  # Only update history if the model actually saw something
                    if ema_kpts[i] is None:
                        ema_kpts[i] = kpts[i].copy()
                    else:
                        ema_kpts[i] = args.ema_alpha * kpts[i] + (1 - args.ema_alpha) * ema_kpts[i]
                    
                    # Override the raw vibrating pixel coordinate with the smoothly gliding coordinate
                    kpts[i] = ema_kpts[i]

            # Use higher threshold for per-keypoint confidence
            kp_threshold = args.kp_conf
            valid_mask = kconf > kp_threshold
            n_valid = int(np.sum(valid_mask))

            if n_valid >= 3:
                keypoints_2d = kpts[valid_mask][:n_valid].astype(np.float64)
                keypoints_3d = PITCH_CORNERS_3D[valid_mask][:n_valid]

        # 4. Initialize or update tracker
        if tracker is None or not tracker.initialized:
            # Need initialization
            if keypoints_2d is not None and n_valid >= 4:
                tracker = CameraTracker(w, h, tripod_position=tripod_position, tripod_radius=tripod_radius)
                success = tracker.reinit(keypoints_2d, keypoints_3d)
                if success:
                    mode = "INIT"
                    consecutive_failures = 0
                    info = tracker.get_camera_info()
                    print(f"  [Frame {count}] INITIALIZED: Z={info['position'][2]:.1f}m, "
                          f"hfov={info['hfov_deg']:.0f}°")
                    
                    # Also seed optical flow immediately
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
            # Tracker exists — do update
            yolo_succeeded = keypoints_2d is not None and (n_valid >= 3 or (args.use_predictive_anchors and n_valid > 0))

            if yolo_succeeded:
                reproj_err, new_params = tracker.update(
                    keypoints_2d, keypoints_3d,
                    rotation_weight=args.rotation_penalty,
                    edge_points=edge_points,
                    use_lines=args.use_lines,
                    use_predictive_anchors=args.use_predictive_anchors
                )

                if reproj_err is not None:
                    mode = "OPTIMIZED"
                    consecutive_failures = 0
                    if count % 10 == 0:
                        info = tracker.get_camera_info()
                        print(f"  [Frame {count}] OPTIMIZED: reproj={reproj_err:.1f}px, "
                              f"Z={info['position'][2]:.1f}m, hfov={info['hfov_deg']:.0f}°")

                    # Refresh optical flow features on every successful YOLO frame
                    if args.use_optical_flow and tracker.params is not None:
                        of_pts, of_3d = extract_of_features(
                            gray, tracker.params, tracker.principal_point, w, h,
                            yaw_flip=tracker.yaw_flip
                        )
                        if of_pts is not None and count % 30 == 0:
                            print(f"  [Frame {count}] OF refreshed: {len(of_pts)} features")
                else:
                    mode = "OPT_REJECTED"
                    consecutive_failures += 1
                    if count % 10 == 0:
                        print(f"  [Frame {count}] OPT REJECTED")

            elif args.use_optical_flow and of_pts is not None and prev_gray is not None and tracker.params is not None:
                # YOLO failed — try optical flow bridging
                new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, of_pts, None, **LK_PARAMS
                )
                good = status.flatten() == 1
                if np.sum(good) >= MIN_OF_FEATURES:
                    of_pts = new_pts[good].reshape(-1, 1, 2)
                    of_3d = of_3d[good]

                    flow_2d = of_pts.reshape(-1, 2).astype(np.float64)
                    flow_3d = of_3d.astype(np.float64)

                    # backproject_to_ground returns 3D in the tracker's
                    # internal frame (already flipped when yaw_flip=True).
                    # tracker.update() will flip again, so undo the flip here
                    # so the net result is a single flip inside update().
                    if tracker.yaw_flip:
                        flow_3d = flow_3d.copy()
                        flow_3d[:, 0] *= -1
                        flow_3d[:, 1] *= -1

                    reproj_err, new_params = tracker.update(
                        flow_2d, flow_3d,
                        rotation_weight=5.0, # Flow provides many reliable points, no need for massive stubborness
                        edge_points=edge_points,
                        use_lines=args.use_lines,
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
                        if count % 10 == 0:
                            print(f"  [Frame {count}] FLOW REJECTED")
                else:
                    mode = "FLOW_LOST"
                    consecutive_failures += 1
                    of_pts = None
                    of_3d = None
                    if count % 30 == 0:
                        print(f"  [Frame {count}] FLOW LOST ({int(np.sum(good))} features left)")

            else:
                mode = "COAST"
                consecutive_failures += 1
                if count % 30 == 0:
                    print(f"  [Frame {count}] COASTING ({n_valid} keypoints)")

            # Re-initialize if the tracker has been failing for too long.
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
            elif mode == "FLOW":
                frame_entry["score"] = 0.8
                frame_entry["reproj_error_px"] = float(reproj_err) if reproj_err is not None else None
            elif mode == "COAST":
                frame_entry["score"] = 0.1

        # 6. Visualization
        if args.visualize:
            # Draw raw YOLO keypoints
            if r.keypoints is not None and len(r.keypoints.xy) > 0 and r.keypoints.xy.shape[1] >= 8:
                all_kpts = r.keypoints.xy[0].cpu().numpy()
                all_kconf = r.keypoints.conf[0].cpu().numpy() if r.keypoints.conf is not None else np.ones(8)
                vis = draw_keypoints(frame, all_kpts, all_kconf, args.kp_conf, mode)
            else:
                vis = frame.copy()

                
            # Draw Pitch Edges
            if len(edge_points) > 1:
                pts = np.array(edge_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
            for (px, py) in edge_points:
                cv2.circle(vis, (int(px), int(py)), 4, (0, 255, 255), -1)
                
            cv2.putText(vis, f"Frame: {count} | {mode}", (30, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imwrite(os.path.join(args.vis_dir, f"{count+1:06d}.jpg"), vis)

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

    print(f"\n{'='*60}")
    print(f"Saved {len(frames_data)} frames to {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()