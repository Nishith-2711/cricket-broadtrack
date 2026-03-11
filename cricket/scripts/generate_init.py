import cv2
import json
import argparse
import numpy as np
import os
import sys
from ultralytics import YOLO
import math

# Add parent dir to path to import if needed, but we use local config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cricket.src import config


def estimate_pose_pnp(image_points, object_points, frame_width, frame_height, prev_rvec=None, prev_tvec=None, hfov_deg=60.0):
    """
    Solve PnP to find camera pose.
    - Uses IPPE for 4 points (robust for planes).
    - Uses ITERATIVE for < 4 points (needs previous guess).
    """
    focal_length = (frame_width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    center = (frame_width / 2.0, frame_height / 2.0)
    K = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.zeros((4, 1))

    # Select parameters based on point count
    n_points = image_points.shape[0]
    
    if n_points == 4:
         # For ITERATIVE, using a guess helps temporal consistency significantly.
        if prev_rvec is not None and prev_tvec is not None:
             success, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist_coeffs, 
                                                rvec=prev_rvec, tvec=prev_tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        else:
             # First frame or lost tracking: Standard solve
             success, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist_coeffs, flags=cv2.SOLVEPNP_IPPE)
    elif n_points == 3 and prev_rvec is not None and prev_tvec is not None:
        # Fallback: 3 points with guess
        # We need to filter object_points to match the 3 visible image_points.
        # But wait, we passed ALL object points. 
        # The caller needs to filter object_points begenerate ifore calling this if points are missing.
        # For now, let's assume the caller handles matching.
        success, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist_coeffs, 
                                            rvec=prev_rvec, tvec=prev_tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    else:
        return False, None, None, K
    
    return success, rvec, tvec, K

def rvec_tvec_to_broadtrack_params(rvec, tvec, frame_width, frame_height, K):
    """
    Convert OpenCV rvec/tvec into BroadTrack-style parameters.
    This matches BroadTrack's camera model:

        R = (Rpan @ Rtilt @ Rroll).T
        X_cam = R @ (X_world - C)

    """

    # ---------------------------------------------------------
    # 1️⃣ Convert Rodrigues to Rotation Matrix
    # ---------------------------------------------------------
    R_cv, _ = cv2.Rodrigues(rvec)

    # ---------------------------------------------------------
    # 2️⃣ Convert OpenCV to BroadTrack Convention
    # OpenCV: X_cam = R_cv @ X + t
    # BroadTrack: X_cam = R_bt @ (X - C)
    #
    # Relationship:
    #   R_bt = R_cv
    #   C = -R_cv.T @ t
    # ---------------------------------------------------------
    C = -R_cv.T @ tvec.reshape(3,1)

    # ---------------------------------------------------------
    # 3️⃣ Fix Forward Axis Convention
    # OpenCV camera looks along +Z
    # BroadTrack camera looks along -Z
    # ---------------------------------------------------------
    R_bt = R_cv.copy()
    R_bt[:,2] *= -1   # flip forward axis

    # ---------------------------------------------------------
    # 4️⃣ Extract Pan / Tilt / Roll
    #
    # BroadTrack defines:
    #   R = (Rpan @ Rtilt @ Rroll).T
    #
    # So:
    #   R.T = Rpan @ Rtilt @ Rroll
    #
    # Solve from R_bt
    # ---------------------------------------------------------

    R_transpose = R_bt.T

    # Tilt (rotation around X)
    tilt = math.atan2(R_transpose[2,1], R_transpose[2,2])

    # Pan (rotation around Z)
    pan = math.atan2(R_transpose[1,0], R_transpose[0,0])

    # Roll (second Z rotation)
    roll = math.atan2(R_transpose[0,1], R_transpose[0,0])

    # ---------------------------------------------------------
    # 5️⃣ Compute HFOV from focal length
    # ---------------------------------------------------------
    focal = K[0,0]
    hfov = 2 * math.atan((frame_width / 2.0) / focal)
    hfov_deg = math.degrees(hfov)

    # ---------------------------------------------------------
    # 6️⃣ Enforce Physical Camera Constraints
    # Camera should be above ground → Z negative
    # ---------------------------------------------------------
    C[2] = -abs(C[2])

    # ---------------------------------------------------------
    # 7️⃣ Return BroadTrack-Compatible Dictionary
    # ---------------------------------------------------------
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
    prev_points = None
    prev_valid_params = None
    prev_rvec = None
    prev_tvec = None
    smoother = PoseSmoother(alpha=0.5)

    # Constants for tracking
    LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    
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
            # Use abspath to correctly find image
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
                "horizontalFieldOfViewDegrees": 30.0, # Guess
                "normalizedRadialDistortionCoefficients": [0.0]
            },
            "score": 0.0,
            "time": float(count)
        }

        # 2. Try YOLO Detection
        results = model(frame, conf=args.conf, verbose=False)
        r = results[0]
        
        detected_points = None
        confs = None
        detected_indices = [] # Which corners did we find?

        if r.keypoints is not None and len(r.keypoints.xy) > 0 and r.keypoints.xy.shape[1] >= 4:
            kpts = r.keypoints.xy[0].cpu().numpy()
            kconf = r.keypoints.conf[0].cpu().numpy() if r.keypoints.conf is not None else np.ones(4)
            
            # Filter by confidence
            valid_mask = kconf > args.conf
            if np.sum(valid_mask) >= 3:
                 detected_points = kpts[valid_mask].astype(np.float32)
                 confs = kconf[valid_mask]
                 detected_indices = np.where(valid_mask)[0]
        
        # 3. Fallback: Optical Flow Tracking (only if YOLO completely lost OR we have prior 4 points to track)
        # Note: Optical flow usually tracks ALL previous points.
        if is_video and detected_points is None and prev_points is not None and prev_gray is not None:
             p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_grey, prev_points, None, **LK_PARAMS)
             
             # Keep only good points
             good_new = p1[st[:,0] == 1]
             
             if len(good_new) >= 3: # Need at least 3 points to solve
                 detected_points = good_new
                 # We assume indices match previous frame (0,1,2,3)
                 detected_indices = [0,1,2,3][:len(good_new)] 
                 confs = np.array([0.4]*len(good_new))
                 print(f"  [Frame {count}] Tracking {len(good_new)} points")
        
        # 4. Solve PnP & update state
        if detected_points is not None and len(detected_points) >= 3:
            
            # Select corresponding object points
            obj_pts = config.OBJECT_POINTS[detected_indices]
            
            # If processing a folder of images, they might not be continuous frames.
            # Force independent processing by ignoring previous frame.
            if not is_video:
                prev_rvec = None
                prev_tvec = None
            
            # Solve PnP
            # First frame or lost tracking: Standard solve (IPPE for 4 pts)
            if prev_rvec is None:
                flags = cv2.SOLVEPNP_IPPE if len(detected_points) == 4 else cv2.SOLVEPNP_ITERATIVE
                success, rvec, tvec, K = estimate_pose_pnp(detected_points, obj_pts, w, h, None, None, hfov_deg=30.0)
                
                if success:
                    # Check Camera Position X. If negative (Batting End), flip it.
                    # R_cv, _ = cv2.Rodrigues(rvec)
                    # C = -R_cv.T @ tvec
                    # We can use our helper function to get C
                    temp_params = rvec_tvec_to_broadtrack_params(rvec, tvec, w, h, K)
                    
                    if temp_params["positionXMeters"] < 0:
                        print(f"  [Frame {count}] Initial solution inverted (X={temp_params['positionXMeters']:.2f}m). Flipping to Bowling End...")
                        R_old, _ = cv2.Rodrigues(rvec)
                        R_z_180 = np.array([
                            [-1, 0, 0],
                            [0, -1, 0],
                            [0, 0, 1]
                        ], dtype=np.float32)
                        
                        R_new = R_old @ R_z_180
                        rvec, _ = cv2.Rodrigues(R_new)

            else:
                 # Use previous guess
                success, rvec, tvec, K = estimate_pose_pnp(detected_points, obj_pts, w, h, prev_rvec, prev_tvec)
            
            if success:
                # Temporal Consistency Check
                is_valid_solution = True
                
                if prev_tvec is not None:
                    dist = np.linalg.norm(tvec - prev_tvec)
                    # Relax check slightly for 3-point solutions which might be noisier
                    thresh = 8.0 if len(detected_points) < 4 else 5.0
                    if dist > thresh: 
                         print(f"  [Frame {count}] REJECTED jump: {dist:.2f}m")
                         is_valid_solution = False
                
                if is_valid_solution:
                    # Smooth
                    if is_video:
                        smooth_rvec, smooth_tvec = smoother.update(rvec, tvec)
                    else:
                        smooth_rvec, smooth_tvec = rvec, tvec
                    
                    # Save valid parameters
                    params = rvec_tvec_to_broadtrack_params(smooth_rvec, smooth_tvec, w, h, K)
                    params["positionZMeters"] = -abs(params["positionZMeters"]) # Force Z < 0 (above ground)
                    
                    frame_entry["cp"].update(params)
                    frame_entry["score"] = float(np.mean(confs))
                    print(f"  [Frame {count}] Accepted Z: {params['positionZMeters']:.2f}m")
                    
                    # Update History
                    prev_points = detected_points 
                    # NOTE: For flow tracking to work next frame, we ideally want 4 points.
                    # If we only have 3, we track 3.
                    
                    prev_rvec = smooth_rvec
                    prev_tvec = smooth_tvec
                    prev_valid_params = params
        
        # 5. Last Resort: Coasting
        if frame_entry["score"] == 0.0 and prev_valid_params is not None:
             frame_entry["cp"].update(prev_valid_params)
             frame_entry["score"] = 0.1

        # Store entry
        frames_data[key] = frame_entry
        
        # Update previous frame for next iteration
        prev_gray = frame_grey.copy()

        count += 1
        if count % 100 == 0:
            print(f"Processed {count}/{total_frames}")

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
