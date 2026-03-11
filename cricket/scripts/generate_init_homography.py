"""
Homography-Based Cricket Pitch Tracking

Instead of PnP (4 points -> 6-DOF pose, ill-conditioned for planes):
1. Detect 4 corners with YOLO -> compute Homography H (pitch plane -> image)
2. Track scene features with LK optical flow -> compute frame-to-frame homography
3. Accumulate: H_new = H_delta @ H_old
4. Project ALL pitch geometry through H — no hallucinations possible

The homography is the mathematically exact transformation for planar scenes.
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
from cricket.src import config

# ─────────────────────────────────────────────────────
# PITCH GEOMETRY (2D on Z=0 plane, in meters)
# ─────────────────────────────────────────────────────

L = config.PITCH_LENGTH / 2.0  # 10.06m
W = config.PITCH_WIDTH / 2.0   # 1.525m

PITCH_CORNERS_2D = np.array([
    [-L, -W],  # Batting Left
    [-L,  W],  # Batting Right
    [ L,  W],  # Bowling Right
    [ L, -W],  # Bowling Left
], dtype=np.float32)

POP_DIST = 1.22
CREASE_LINES = [
    ([-L + POP_DIST, -W], [-L + POP_DIST, W]),
    ([L - POP_DIST, -W], [L - POP_DIST, W]),
    ([-L, -W], [-L, W]),
    ([L, -W], [L, W]),
]

STUMP_POSITIONS = [[-L, 0], [L, 0]]

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)


# ─────────────────────────────────────────────────────
# HOMOGRAPHY UTILITIES
# ─────────────────────────────────────────────────────

def compute_homography_from_corners(image_points, pitch_points=PITCH_CORNERS_2D):
    """Compute H: pitch plane -> image pixels."""
    H, mask = cv2.findHomography(pitch_points, image_points, cv2.RANSAC, 5.0)
    return H


def project_point(H, world_pt):
    """Project a 2D world point to image pixel via H."""
    pt = np.array([world_pt[0], world_pt[1], 1.0])
    proj = H @ pt
    if abs(proj[2]) < 1e-8:
        return None
    return (int(proj[0] / proj[2]), int(proj[1] / proj[2]))


def project_point_float(H, world_pt):
    """Project with float precision (for validation)."""
    pt = np.array([world_pt[0], world_pt[1], 1.0])
    proj = H @ pt
    if abs(proj[2]) < 1e-8:
        return None
    return (proj[0] / proj[2], proj[1] / proj[2])


def validate_homography(H, frame_w, frame_h):
    """
    Check if H projects pitch corners into a valid convex quadrilateral.
    Returns True if the projected corners are geometrically reasonable.
    """
    corners = [project_point_float(H, c) for c in PITCH_CORNERS_2D]
    
    # All corners must project successfully
    if any(c is None for c in corners):
        return False
    
    # At least some corners should be near the frame (within 2x frame size)
    margin = max(frame_w, frame_h) * 2
    in_range = 0
    for x, y in corners:
        if -margin < x < frame_w + margin and -margin < y < frame_h + margin:
            in_range += 1
    if in_range < 2:
        return False
    
    # Check convexity: cross products of consecutive edges should all have same sign
    pts = np.array(corners, dtype=np.float64)
    signs = []
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        p3 = pts[(i + 2) % 4]
        edge1 = p2 - p1
        edge2 = p3 - p2
        cross = edge1[0] * edge2[1] - edge1[1] * edge2[0]
        signs.append(cross)
    
    # All cross products should have the same sign (convex polygon)
    all_pos = all(s > 0 for s in signs)
    all_neg = all(s < 0 for s in signs)
    if not (all_pos or all_neg):
        return False
    
    # Check area is reasonable (not collapsed)
    area = 0.5 * abs(
        (pts[0][0] * pts[1][1] - pts[1][0] * pts[0][1]) +
        (pts[1][0] * pts[2][1] - pts[2][0] * pts[1][1]) +
        (pts[2][0] * pts[3][1] - pts[3][0] * pts[2][1]) +
        (pts[3][0] * pts[0][1] - pts[0][0] * pts[3][1])
    )
    
    frame_area = frame_w * frame_h
    if area < frame_area * 0.001 or area > frame_area * 10:
        return False
    
    return True


def decompose_homography(H, frame_width, frame_height, hfov_deg=30.0):
    """Decompose H into rvec, tvec for init.json compatibility."""
    focal_length = (frame_width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    K = np.array([
        [focal_length, 0, frame_width / 2.0],
        [0, focal_length, frame_height / 2.0],
        [0, 0, 1]
    ], dtype=np.float64)

    K_inv = np.linalg.inv(K)
    M = K_inv @ H
    lam = np.linalg.norm(M[:, 0])
    if lam < 1e-10:
        return None

    M = M / lam
    r1, r2, t = M[:, 0], M[:, 1], M[:, 2]
    r3 = np.cross(r1, r2)

    R_approx = np.column_stack([r1, r2, r3])
    U, _, Vt = np.linalg.svd(R_approx)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = -R
        t = -t

    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)
    return rvec, tvec, K


def homography_to_init_params(H, frame_width, frame_height):
    """Convert homography to init.json camera parameters."""
    result = decompose_homography(H, frame_width, frame_height)
    if result is None:
        return None

    rvec, tvec, K = result
    R, _ = cv2.Rodrigues(rvec)
    C = -R.T @ tvec

    focal = K[0, 0]
    hfov = 2 * math.atan((frame_width / 2.0) / focal)
    C[2, 0] = -abs(C[2, 0])

    return {
        "panDegrees": 0.0,
        "tiltDegrees": 0.0,
        "rollDegrees": 0.0,
        "positionXMeters": float(C[0, 0]),
        "positionYMeters": float(C[1, 0]),
        "positionZMeters": float(C[2, 0]),
        "horizontalFieldOfViewDegrees": math.degrees(hfov),
        "normalizedRadialDistortionCoefficients": [0.0],
        "rvec": rvec.flatten().tolist(),
        "tvec": tvec.flatten().tolist(),
        "homography": H.tolist()
    }


# ─────────────────────────────────────────────────────
# CAMERA CUT DETECTION
# ─────────────────────────────────────────────────────

def detect_camera_cut(prev_gray, curr_gray, threshold=0.4):
    """
    Detect a camera cut by comparing histograms.
    Returns True if a cut is detected.
    """
    hist1 = cv2.calcHist([prev_gray], [0], None, [64], [0, 256])
    hist2 = cv2.calcHist([curr_gray], [0], None, [64], [0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return score < threshold  # Low correlation = camera cut


# ─────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────

def draw_pitch_from_homography(img, H, mode=""):
    """Draw pitch lines, creases, stumps using H."""
    vis = img.copy()
    h, w = vis.shape[:2]

    def ok(pt, margin=500):
        return pt is not None and -margin <= pt[0] < w + margin and -margin <= pt[1] < h + margin

    corners_px = [project_point(H, c) for c in PITCH_CORNERS_2D]
    for i in range(4):
        p1, p2 = corners_px[i], corners_px[(i + 1) % 4]
        if ok(p1) and ok(p2):
            cv2.line(vis, p1, p2, config.COLOR_PITCH, 2)

    for start, end in CREASE_LINES:
        p1, p2 = project_point(H, start), project_point(H, end)
        if ok(p1) and ok(p2):
            cv2.line(vis, p1, p2, config.COLOR_CREASE, 2)

    for sp in STUMP_POSITIONS:
        pt = project_point(H, sp)
        if ok(pt):
            cv2.circle(vis, pt, 5, config.COLOR_STUMP, -1)

    labels = ["BatL", "BatR", "BowR", "BowL"]
    for corner, label in zip(corners_px, labels):
        if ok(corner):
            cv2.circle(vis, corner, 6, (0, 0, 255), -1)
            cv2.putText(vis, label, (corner[0] + 10, corner[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return vis


# ─────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Homography-based pitch tracking")
    parser.add_argument('-v', '--video', help='Path to input video')
    parser.add_argument('-i', '--images', help='Path to input image folder')
    parser.add_argument('-m', '--model', required=True, help='Path to YOLO pose model')
    parser.add_argument('-o', '--output', required=True, help='Path to output init.json')
    parser.add_argument('--conf', type=float, default=0.25, help='YOLO confidence threshold')
    parser.add_argument('--visualize', action='store_true', help='Save visualization frames')
    parser.add_argument('--vis-dir', default='homography_vis', help='Visualization output dir')
    parser.add_argument('--reinit-interval', type=int, default=30,
                       help='Re-detect with YOLO every N frames to correct drift')
    args = parser.parse_args()

    if not args.video and not args.images:
        print("Error: Must provide either --video or --images")
        return

    model = YOLO(args.model)
    
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
        frames_source = sorted([f for f in os.listdir(args.images) if os.path.splitext(f)[1].lower() in exts])
        total_frames = len(frames_source)
        print(f"Found {total_frames} images in {args.images}")

    if args.visualize:
        os.makedirs(args.vis_dir, exist_ok=True)

    # ── STATE ──
    H_current = None
    prev_gray = None
    prev_features = None
    frames_data = {}
    count = 0
    frames_since_yolo = 0
    consecutive_track_failures = 0
    MIN_FEATURES = 15

    print(f"\n{'='*60}")
    print("HOMOGRAPHY-BASED PITCH TRACKING (LK Optical Flow)")
    print(f"{'='*60}")
    print(f"YOLO re-init every: {args.reinit_interval} frames")

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

        # ── 2. Camera cut detection ──
        camera_cut = False
        if prev_gray is not None:
            camera_cut = detect_camera_cut(prev_gray, gray)
            if camera_cut:
                print(f"  [Frame {count}] CAMERA CUT DETECTED — re-initializing")
                H_current = None
                prev_features = None
                consecutive_track_failures = 0

        # ── 3. Decide: track or detect? ──
        need_yolo = (
            H_current is None or
            camera_cut or
            frames_since_yolo >= args.reinit_interval or
            consecutive_track_failures >= 3
        )

        # ── 4A. LK Optical Flow Tracking ──
        if not need_yolo and prev_gray is not None and prev_features is not None and len(prev_features) >= MIN_FEATURES:
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_features, None, **LK_PARAMS)

            good_mask = st[:, 0] == 1
            pts_old = prev_features[good_mask]
            pts_new = p1[good_mask]

            if len(pts_old) >= MIN_FEATURES:
                H_delta, mask = cv2.findHomography(
                    pts_old.reshape(-1, 2), pts_new.reshape(-1, 2), cv2.RANSAC, 3.0
                )

                if H_delta is not None:
                    # Normalize
                    H_delta = H_delta / H_delta[2, 2]
                    
                    # Compute candidate H
                    H_candidate = H_delta @ H_current
                    
                    # Validate: do projected corners form a valid convex quad?
                    is_valid = validate_homography(H_candidate, w, h)
                    if is_valid:
                        H_current = H_candidate
                        mode = "TRACKED"
                        consecutive_track_failures = 0
                        
                        # Keep inlier features
                        if mask is not None:
                            inlier_mask = mask.ravel().astype(bool)
                            prev_features = pts_new[inlier_mask].reshape(-1, 1, 2)
                        else:
                            prev_features = pts_new.reshape(-1, 1, 2)
                        
                        # Re-detect features if running low
                        if len(prev_features) < 50:
                            new_feats = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=10)
                            if new_feats is not None:
                                prev_features = new_feats
                        
                        n_inliers = int(mask.sum()) if mask is not None else len(pts_new)
                        if count % 5 == 0:
                            print(f"  [Frame {count}] TRACKED ({n_inliers} inliers, {len(prev_features)} features)")
                    else:
                        consecutive_track_failures += 1
                        print(f"  [Frame {count}] TRACK_INVALID (bad quad, failures={consecutive_track_failures})")
                else:
                    consecutive_track_failures += 1
                    print(f"  [Frame {count}] H_delta is None (failures={consecutive_track_failures})")
            else:
                consecutive_track_failures += 1
                need_yolo = True
                print(f"  [Frame {count}] TOO_FEW_FEATURES ({len(pts_old)} tracked, failures={consecutive_track_failures})")
        elif not need_yolo:
            reasons = []
            if prev_gray is None: reasons.append("no prev_gray")
            if prev_features is None: reasons.append("no prev_features")
            elif len(prev_features) < MIN_FEATURES: reasons.append(f"only {len(prev_features)} features")
            if count < 10:
                print(f"  [Frame {count}] SKIP_TRACK: {', '.join(reasons)}")

        # ── 4B. YOLO Detection → Compute H ──
        if mode == "NONE" and need_yolo:
            results = model(frame, conf=args.conf, verbose=False)
            r = results[0]

            detected = False
            if r.keypoints is not None and len(r.keypoints.xy) > 0 and r.keypoints.xy.shape[1] >= 4:
                kpts = r.keypoints.xy[0].cpu().numpy()
                kconf = r.keypoints.conf[0].cpu().numpy() if r.keypoints.conf is not None else np.ones(4)

                valid_mask = kconf > args.conf
                n_valid = int(np.sum(valid_mask))

                if n_valid >= 4:
                    image_pts = kpts[valid_mask][:4].astype(np.float32)
                    pitch_pts = PITCH_CORNERS_2D[valid_mask][:4]
                    H_new = compute_homography_from_corners(image_pts, pitch_pts)

                    if H_new is not None and validate_homography(H_new, w, h):
                        H_current = H_new
                        mode = "YOLO"
                        frames_since_yolo = 0
                        consecutive_track_failures = 0
                        detected = True

                        prev_features = cv2.goodFeaturesToTrack(
                            gray, maxCorners=200, qualityLevel=0.01, minDistance=10
                        )
                        n_feats = len(prev_features) if prev_features is not None else 0
                        print(f"  [Frame {count}] YOLO -> H ({n_valid} pts, {n_feats} features)")
            
            if not detected and H_current is not None:
                mode = "COAST"
                if count % 30 == 0:
                    print(f"  [Frame {count}] COASTING (YOLO failed)")

        # ── 5. Store results ──
        if H_current is not None:
            params = homography_to_init_params(H_current, w, h)
            if params is not None:
                frame_entry["cp"].update(params)
                if mode == "YOLO":
                    frame_entry["score"] = 1.0
                elif mode == "TRACKED":
                    frame_entry["score"] = 0.8
                elif mode == "COAST":
                    frame_entry["score"] = 0.1
                else:
                    frame_entry["score"] = 0.1

        # ── 6. Visualization ──
        if args.visualize and H_current is not None:
            vis = draw_pitch_from_homography(frame, H_current, mode)
            cv2.putText(vis, f"Frame: {count} | {mode}", (30, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imwrite(os.path.join(args.vis_dir, f"{count+1:06d}.jpg"), vis)

        frames_data[key] = frame_entry
        prev_gray = gray.copy()
        frames_since_yolo += 1
        count += 1

        if count % 100 == 0:
            print(f"  Processed {count}/{total_frames} frames...")

    if is_video:
        cap.release()

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
