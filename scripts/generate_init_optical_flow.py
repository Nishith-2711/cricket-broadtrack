import cv2
import json
import argparse
import numpy as np
import os
import sys
from ultralytics import YOLO
import math

# Add parent dir to path to import if needed, but we use local config

from broadtrack import config

def estimate_pose_pnp(image_points, object_points, frame_width, frame_height, prev_rvec=None, prev_tvec=None, hfov_deg=60.0):
    """
    Solve PnP to find camera pose.
    - Uses IPPE for 4 points (robust for planes).
    - Uses ITERATIVE for < 4 points (needs previous guess) or when we have many points.
    """
    focal_length = (frame_width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    center = (frame_width / 2.0, frame_height / 2.0)
    K = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.zeros((4, 1))

    n_points = image_points.shape[0]
    
    if n_points >= 4:
         if prev_rvec is not None and prev_tvec is not None:
             success, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist_coeffs, 
                                                rvec=prev_rvec, tvec=prev_tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
         else:
             # Fast initial solve
             if n_points == 4:
                 success, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist_coeffs, flags=cv2.SOLVEPNP_IPPE)
             else:
                 success, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
    elif n_points == 3 and prev_rvec is not None and prev_tvec is not None:
        success, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist_coeffs, 
                                            rvec=prev_rvec, tvec=prev_tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    else:
        return False, None, None, K
    return success, rvec, tvec, K

def rvec_tvec_to_broadtrack_params(rvec, tvec, frame_width, frame_height, K):
    R_cv, _ = cv2.Rodrigues(rvec)
    C = -R_cv.T @ tvec.reshape(3,1)

    R_bt = R_cv.copy()
    R_bt[:,2] *= -1   # flip forward axis

    R_transpose = R_bt.T

    # Tilt (rotation around X)
    tilt = math.atan2(R_transpose[2,1], R_transpose[2,2])

    # Pan (rotation around Z)
    pan = math.atan2(R_transpose[1,0], R_transpose[0,0])

    # Roll (second Z rotation)
    roll = math.atan2(R_transpose[0,1], R_transpose[0,0])

    focal = K[0,0]
    hfov = 2 * math.atan((frame_width / 2.0) / focal)
    hfov_deg = math.degrees(hfov)

    C[2] = -abs(C[2])

    return {
        "panDegrees": math.degrees(pan),
        "tiltDegrees": math.degrees(tilt),
        "rollDegrees": math.degrees(roll),
        "positionXMeters": float(C[0].item()),
        "positionYMeters": float(C[1].item()),
        "positionZMeters": float(C[2].item()),
        "horizontalFieldOfViewDegrees": hfov_deg,
        "normalizedRadialDistortionCoefficients": [0.0],
        "rvec": rvec.flatten().tolist(),
        "tvec": tvec.flatten().tolist()
    }


class PoseSmoother:
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.smooth_rvec = None
        self.smooth_tvec = None

    def update(self, rvec, tvec):
        if self.smooth_rvec is None:
            self.smooth_rvec = rvec
            self.smooth_tvec = tvec
        else:
            self.smooth_rvec = self.alpha * rvec + (1 - self.alpha) * self.smooth_rvec
            self.smooth_tvec = self.alpha * tvec + (1 - self.alpha) * self.smooth_tvec
        return self.smooth_rvec, self.smooth_tvec

def get_3d_points_from_image_points(image_points, rvec, tvec, K):
    """
    Given a camera pose (rvec, tvec, K) and 2D image points on a Z=0 plane (pitch),
    compute their 3D object coordinates in the world.
    """
    R, _ = cv2.Rodrigues(rvec)
    
    # We want to solve for X, Y given Z=0
    # s [u, v, 1]^T = K (R [X, Y, 0]^T + t)
    # s K^-1 [u, v, 1]^T = R_1 X + R_2 Y + t
    # s K^-1 [u, v, 1]^T - t = [R_1 R_2] [X, Y]^T
    
    K_inv = np.linalg.inv(K)
    obj_pts = []
    
    for pt in image_points:
        uv1 = np.array([[pt[0]], [pt[1]], [1.0]])
        ray_cam = K_inv @ uv1
        ray_world = R.T @ ray_cam
        cam_pos = -R.T @ tvec
        
        # Intersection with Z=0 plane
        # cam_pos.z + s * ray_world.z = 0  =>  s = -cam_pos.z / ray_world.z
        if abs(ray_world[2,0]) > 1e-6:
            s_val = -cam_pos[2,0] / ray_world[2,0]
            intersection = cam_pos + s_val * ray_world
            obj_pts.append([intersection[0,0], intersection[1,0], 0.0])
        else:
            obj_pts.append([0.0, 0.0, 0.0]) # Fallback (shouldn't happen for valid rays)
            
    return np.array(obj_pts, dtype=np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', help='Path to input video')
    parser.add_argument('-i', '--images', help='Path to input image folder')
    parser.add_argument('-m', '--model', required=True, help='Path to YOLO pose model')
    parser.add_argument('-o', '--output', required=True, help='Path to output init.json')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    
    args = parser.parse_args()
    
    if not args.video and not args.images:
        print("Error: Must provide either --video or --images")
        return

    # Load Model
    model = YOLO(args.model)
    
    frames_source = []
    is_video = False
    
    if args.video:
        is_video = True
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"Error: Cannot open video {args.video}")
            return
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        # Images
        exts = ['.jpg', '.jpeg', '.png', '.bmp']
        files = sorted([f for f in os.listdir(args.images) if os.path.splitext(f)[1].lower() in exts])
        frames_source = files
        total_frames = len(files)
        print(f"Found {total_frames} images in {args.images}")

    frames_data = {}
    count = 0
    
    # State tracking
    prev_gray = None
    prev_valid_params = None
    prev_rvec = None
    prev_tvec = None
    smoother = PoseSmoother(alpha=0.5)

    # Optical Flow State
    # List of active 2D points being tracked
    tracked_points = None
    # Corresponding 3D world coordinates for the tracked points
    tracked_obj_points = None
    
    # Constants for tracking
    LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    
    MIN_FEATURES_TO_TRACK = 10
    
    # Process
    while True:
        # 1. Read Frame
        if is_video:
            ret, frame = cap.read()
            if not ret:
                break
            filename = f"frame_{count:06d}.jpg" 
        else:
            if count >= total_frames:
                break
            filename = frames_source[count]
            image_path = os.path.abspath(os.path.join(args.images, filename))
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Warning: Could not read {image_path}")
                count += 1
                continue
        
        frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]
        key = f"/workspace/cricket/{filename}"

        # Initialize frame entry
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

        mode = "COAST"
        detected_points = None
        confs = None
        
        # Try finding points with Optical Flow first if we have enough tracked features
        if is_video and tracked_points is not None and prev_gray is not None and len(tracked_points) >= MIN_FEATURES_TO_TRACK:
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_grey, tracked_points, None, **LK_PARAMS)
            
            good_mask = st[:, 0] == 1
            good_new = p1[good_mask]
            
            if len(good_new) >= MIN_FEATURES_TO_TRACK:
                mode = "FLOW"
                tracked_points = good_new
                tracked_obj_points = tracked_obj_points[good_mask]
                
                # Now we solve PnP using the tracked features
                if prev_rvec is not None and prev_tvec is not None:
                     success, rvec, tvec, K = estimate_pose_pnp(tracked_points, tracked_obj_points, w, h, prev_rvec, prev_tvec)
                     
                     if success:
                         # Check for large pose jumps
                         dist = np.linalg.norm(tvec - prev_tvec)
                         if dist > 8.0:
                             print(f"  [Frame {count}] OF REJECTED jump: {dist:.2f}m")
                             success = False
                         
                         # Reprojection error check for optical flow
                         if success:
                             proj_pts, _ = cv2.projectPoints(tracked_obj_points, rvec, tvec, K, np.zeros((4,1)))
                             reproj_err = np.linalg.norm(proj_pts.reshape(-1,2) - tracked_points.reshape(-1,2), axis=1).mean()
                             if reproj_err > 25.0:
                                 print(f"  [Frame {count}] OF REJECTED reproj: {reproj_err:.1f}px")
                                 success = False
                             
                         if success:
                             # We have a pose!
                             smooth_rvec, smooth_tvec = smoother.update(rvec, tvec)
                             params = rvec_tvec_to_broadtrack_params(smooth_rvec, smooth_tvec, w, h, K)
                             params["positionZMeters"] = -abs(params["positionZMeters"])
                             
                             frame_entry["cp"].update(params)
                             frame_entry["score"] = 0.8 # High score for flow
                             print(f"  [Frame {count}] FLOW OK ({len(tracked_points)} pts). Z: {params['positionZMeters']:.2f}m")
                             
                             prev_rvec = smooth_rvec
                             prev_tvec = smooth_tvec
                             prev_valid_params = params
                         else:
                             mode = "FLOW_FAILED"
                             tracked_points = None
                    
            else:
                 mode = "LOST_FEATURES"
                 tracked_points = None
        
        # If we need to re-detect (either flow failed, lost features, or first frame)
        if mode != "FLOW" or tracked_points is None:
            results = model(frame, conf=args.conf, verbose=False)
            r = results[0]
            
            if r.keypoints is not None and len(r.keypoints.xy) > 0 and r.keypoints.xy.shape[1] >= 4:
                kpts = r.keypoints.xy[0].cpu().numpy()
                kconf = r.keypoints.conf[0].cpu().numpy() if r.keypoints.conf is not None else np.ones(4)
                
                valid_mask = kconf > args.conf
                if np.sum(valid_mask) >= 3: # Need at least 3 points to initialize PnP
                     detected_points = kpts[valid_mask].astype(np.float32)
                     confs = kconf[valid_mask]
                     detected_indices = np.where(valid_mask)[0]
                     
                     obj_pts = config.OBJECT_POINTS[detected_indices]
                     
                     if prev_rvec is None:
                         # Try multiple hfov values and pick the best one
                         best_result = None
                         best_error = float('inf')
                         
                         for try_hfov in [30.0, 45.0, 60.0]:
                             s, rv, tv, Kt = estimate_pose_pnp(detected_points, obj_pts, w, h, None, None, hfov_deg=try_hfov)
                             if s:
                                 proj, _ = cv2.projectPoints(obj_pts, rv, tv, Kt, np.zeros((4,1)))
                                 err = np.linalg.norm(proj.reshape(-1,2) - detected_points, axis=1).mean()
                                 
                                 # Validate camera height (Z should be negative = above ground)
                                 tp = rvec_tvec_to_broadtrack_params(rv, tv, w, h, Kt)
                                 if tp["positionZMeters"] > -2.0:
                                     print(f"    hfov={try_hfov}: Z={tp['positionZMeters']:.1f}m (too close to ground), skip")
                                     continue
                                 
                                 print(f"    hfov={try_hfov}: reproj={err:.1f}px, Z={tp['positionZMeters']:.1f}m")
                                 if err < best_error:
                                     best_error = err
                                     best_result = (rv, tv, Kt)
                         
                         if best_result is not None:
                             rvec, tvec, K = best_result
                             success = True
                             temp_params = rvec_tvec_to_broadtrack_params(rvec, tvec, w, h, K)
                             if temp_params["positionXMeters"] < 0:
                                 R_old, _ = cv2.Rodrigues(rvec)
                                 R_z_180 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)
                                 R_new = R_old @ R_z_180
                                 rvec, _ = cv2.Rodrigues(R_new)
                         else:
                             success = False
                     else:
                         success, rvec, tvec, K = estimate_pose_pnp(detected_points, obj_pts, w, h, prev_rvec, prev_tvec)
                         if success:
                             dist = np.linalg.norm(tvec - prev_tvec)
                             if dist > 8.0:
                                 success = False
                                 
                     if success:
                         smooth_rvec, smooth_tvec = smoother.update(rvec, tvec)
                         params = rvec_tvec_to_broadtrack_params(smooth_rvec, smooth_tvec, w, h, K)
                         params["positionZMeters"] = -abs(params["positionZMeters"])
                         
                         frame_entry["cp"].update(params)
                         frame_entry["score"] = float(np.mean(confs))
                         print(f"  [Frame {count}] YOLO DETECTED. Z: {params['positionZMeters']:.2f}m")
                         
                         prev_rvec = smooth_rvec
                         prev_tvec = smooth_tvec
                         prev_valid_params = params
                         
                         # Initialize Good Features to Track for future frames
                         # Mask around the pitch polygon to only extract features on the pitch
                         mask = np.zeros_like(frame_grey)
                         
                         # Project full pitch polygon to image to create mask
                         pitch_corners = config.OBJECT_POINTS
                         pitch_px = []
                         R, _ = cv2.Rodrigues(smooth_rvec)
                         for corner in pitch_corners:
                             c = np.array([[corner[0]], [corner[1]], [0.0]])
                             proj = K @ (R @ c + smooth_tvec)
                             if proj[2,0] > 0.1:
                                 px = int(proj[0,0] / proj[2,0])
                                 py = int(proj[1,0] / proj[2,0])
                                 pitch_px.append([px, py])
                                 
                         if len(pitch_px) == 4:
                             pts = np.array(pitch_px, np.int32).reshape((-1, 1, 2))
                             # slightly inflate the mask to capture the pitch lines itself
                             cv2.fillPoly(mask, [pts], 255)
                             
                             new_features = cv2.goodFeaturesToTrack(frame_grey, mask=mask, maxCorners=100, qualityLevel=0.01, minDistance=10)
                             
                             if new_features is not None:
                                 # We also include the original YOLO corners as features
                                 tracked_points = np.vstack((detected_points.reshape(-1, 1, 2), new_features))
                             else:
                                 tracked_points = detected_points.reshape(-1, 1, 2)
                                 
                             # Now compute the exact 3D location of these new features on the Z=0 plane
                             tracked_obj_points = get_3d_points_from_image_points(tracked_points.reshape(-1,2), smooth_rvec, smooth_tvec, K)
                             print(f"  [Frame {count}] INITIALIZED {len(tracked_points)} features for Optical Flow")

        # 5. Last Resort: Coasting
        if frame_entry["score"] == 0.0 and prev_valid_params is not None:
             frame_entry["cp"].update(prev_valid_params)
             frame_entry["score"] = 0.1
             print(f"  [Frame {count}] COASTING")

        # Store entry
        frames_data[key] = frame_entry
        
        # Update previous frame for next iteration
        prev_gray = frame_grey.copy()

        count += 1

    if is_video:
        cap.release()
    
    # Save JSON
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(frames_data, f, indent=4)
        
    print(f"Detailed init.json saved to {args.output}")

if __name__ == "__main__":
    main()
