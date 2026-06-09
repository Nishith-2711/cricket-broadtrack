import json
import cv2
import numpy as np
import math
import os
import argparse
import sys

# Add parent dir to path to import if needed, but we use local config

from broadtrack import config

def pan_tilt_roll_to_orientation(pan, tilt, roll):
    """
    Converts pan, tilt, roll (in radians) to a rotation matrix.
    BroadTrack convention: R = (Rpan * Rtilt * Rroll).T
    """
    Rpan = np.array([
        [np.cos(pan), -np.sin(pan), 0],
        [np.sin(pan),  np.cos(pan), 0],
        [0, 0, 1]])

    Rroll = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll),  np.cos(roll), 0],
        [0, 0, 1]])

    Rtilt = np.array([l̥
        [1, 0, 0],
        [0, np.cos(tilt), -np.sin(tilt)],
        [0, np.sin(tilt),  np.cos(tilt)]])

    rotMat = Rpan @ Rtilt @ Rroll
    return rotMat.T

def project_point_3d(p_world, R, C):
    """ Transforms point to camera frame. """
    X = np.array(p_world).reshape(3, 1)
    X_cam = R @ (X - C)
    return X_cam

def project_camera_to_pixel(X_cam, K):
    """ Projects camera frame point to pixels. """
    if X_cam[2] <= 0.1: # Near clip plane
        return None
    
    x_proj = K @ X_cam
    x_proj /= x_proj[2]
    return (int(x_proj[0].item()), int(x_proj[1].item()))

def draw_point_3d(img, p, R, C, K, color, radius=5, thickness=-1):
    cam_p = project_point_3d(p, R, C)
    
    # Simple Z clipping
    if cam_p[2] <= 0.1:
        return 0
        
    uv = project_camera_to_pixel(cam_p, K)
    
    if uv:
        # Check if point is actually inside the image frame
        h, w = img.shape[:2]
        if 0 <= uv[0] < w and 0 <= uv[1] < h:
            cv2.circle(img, uv, radius, color, thickness)
            return 1
    return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Path to init.json')
    parser.add_argument('-v', '--video', help='Path to video file')
    parser.add_argument('-f', '--images', help='Path to image folder')
    parser.add_argument('-o', '--output', required=True, help='Path to output folder')
    parser.add_argument('-n', '--limit', type=int, help='Limit number of frames')
    
    args = parser.parse_args()
    
    if not args.video and not args.images:
        print("Error: Must provide either --video or --images")
        return

    # Load JSON
    with open(args.input, 'r') as f:
        data = json.load(f)
        
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
        total_frames = len(files)
        print(f"Found {total_frames} images in {args.images}")
        
    os.makedirs(args.output, exist_ok=True)
    
    # Prepare Geometry from Config
    # Pitch Rectangle
    L_half = config.PITCH_LENGTH / 2.0
    W_half = config.PITCH_WIDTH / 2.0
    
    pitch_corners = [
        (-L_half, -W_half, 0), # Batting Left
        (-L_half,  W_half, 0), # Batting Right
        ( L_half,  W_half, 0), # Bowling Right
        ( L_half, -W_half, 0)  # Bowling Left
    ]
    
    count = 0
    
    while True:
        if is_video:
            ret, img = cap.read()
            if not ret:
                break
            filename = f"{count+1:06d}.jpg" 
        else:
            if count >= total_frames:
                break
            filename = files[count]
            img_path = os.path.join(args.images, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read {img_path}")
                count += 1
                continue
            
        if args.limit and count >= args.limit:
            break
            
        # Construct key using BroadTrack format convention
        if is_video:
            init_filename = f"frame_{count:06d}.jpg"
        else:
            init_filename = filename

        key = f"/workspace/cricket/{init_filename}"
        
        # Fallback search if key not found (in case of different key format)
        if key not in data:
             # Try simple filename match
             for k in data.keys():
                 if os.path.basename(k) == init_filename or os.path.basename(k) == filename:
                     key = k
                     break
        
        if key in data:
            frame_data = data[key]
            cp = frame_data['cp']
            
            # Intrinsics
            h, w = img.shape[:2]
            hfov = math.radians(cp['horizontalFieldOfViewDegrees'])
            fx = (w / 2.0) / math.tan(hfov / 2.0)
            fy = fx
            cx = w / 2.0
            cy = h / 2.0
            
            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
            
            # Extrinsics
            if 'rvec' in cp and 'tvec' in cp:
                rvec = np.array(cp['rvec'])
                tvec = np.array(cp['tvec'])
                R, _ = cv2.Rodrigues(rvec)
                C = -R.T @ tvec.reshape(3,1)
            else:
                pan = math.radians(cp['panDegrees'])
                tilt = math.radians(cp['tiltDegrees'])
                roll = math.radians(cp['rollDegrees'])
                R = pan_tilt_roll_to_orientation(pan, tilt, roll)
                
                C = np.array([
                    cp['positionXMeters'],
                    cp['positionYMeters'],
                    cp['positionZMeters']
                ]).reshape(3, 1)
            
            # Draw
            points_drawn = 0
            is_coasting = frame_data["score"] < 0.2
            
            if is_coasting:
                cv2.putText(img, "COASTING (Low Score)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

            for corner in pitch_corners:
                # Use standard configuration color for points, or a vivid color like red
                if draw_point_3d(img, corner, R, C, K, (0, 0, 255), radius=6, thickness=-1):
                    points_drawn += 1

            if count == 0:
                print(f"DEBUG: Drawn {points_drawn} points on frame {count}")

            cv2.putText(img, f"Frame: {count}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Save exact filename
        outfile = os.path.join(args.output, filename)
        cv2.imwrite(outfile, img)
        
        count += 1
        if count % 100 == 0:
            print(f"Visualized {count} frames...")
            
    if is_video:
        cap.release()
    print("Done point visualization.")

if __name__ == "__main__":
    main()
