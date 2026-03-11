import json
import cv2
import numpy as np
import math
import os
import argparse
import sys

# Add parent dir to path to import if needed, but we use local config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cricket.src import config


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

    Rtilt = np.array([
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
        # print(f"DEBUG: Clipped Z={X_cam[2]}")
        return None
    
    x_proj = K @ X_cam
    x_proj /= x_proj[2]
    return (int(x_proj[0].item()), int(x_proj[1].item()))

def draw_line_3d(img, p1, p2, R, C, K, color, thickness=2):
    cam_p1 = project_point_3d(p1, R, C)
    cam_p2 = project_point_3d(p2, R, C)
    
    # Simple Z clipping
    if cam_p1[2] <= 0.1 or cam_p2[2] <= 0.1:
        # print(f"DEBUG: Z-clip p1={cam_p1.T} p2={cam_p2.T}")
        return 0
        
    uv1 = project_camera_to_pixel(cam_p1, K)
    uv2 = project_camera_to_pixel(cam_p2, K)
    
    if uv1 and uv2:
        # Check if inside image bounds for debugging
        h, w = img.shape[:2]
        if (0 <= uv1[0] < w and 0 <= uv1[1] < h) or (0 <= uv2[0] < w and 0 <= uv2[1] < h):
             pass # Visible
        else:
             # print(f"DEBUG: Line out of bounds: {uv1} -> {uv2}")
             pass
             
        cv2.line(img, uv1, uv2, color, thickness)
        return 1
    return 0
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
    
    # Creases (Popping Crease is 1.22m in front of Bowling Crease)
    pop_dist = 1.22
    
    batting_pop_y = -L_half + pop_dist
    bowling_pop_y =  L_half - pop_dist
    
    # Lines to draw
    lines = []
    
    # Main Pitch Box
    lines.append(([pitch_corners[0], pitch_corners[1]], config.COLOR_PITCH))
    lines.append(([pitch_corners[1], pitch_corners[2]], config.COLOR_PITCH))
    lines.append(([pitch_corners[2], pitch_corners[3]], config.COLOR_PITCH))
    lines.append(([pitch_corners[3], pitch_corners[0]], config.COLOR_PITCH))
    
    # Popping Creases (Across the width)
    # 3.66m wide? (Return crease is 1.22m from center stump? No, 1.32m? 
    # Return creases are "at least 1.22m". Pitch width 3.05m = 1.525m from center.
    # Return creases are usually the width of the pitch?
    # Let's assume Popping Crease spans width of pitch for visualization.
    lines.append(([(-L_half + pop_dist, -W_half, 0), (-L_half + pop_dist, W_half, 0)], config.COLOR_CREASE))
    lines.append(([ (L_half - pop_dist, -W_half, 0), ( L_half - pop_dist, W_half, 0)], config.COLOR_CREASE))
    
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
        # We try to find the key that ends with the filename
        # Or assumes standard key format "/workspace/cricket/{filename}"
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
                # In generates_init: X_cam = R @ X + t
                # project_point_3d expects: X_cam = R_vis @ (X - C)
                # R_vis @ X - R_vis @ C = R @ X + t
                # => R_vis = R
                # => - R_vis @ C = t  => C = - R.T @ t
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
            lines_drawn = 0
            # Check if this is a coasting frame (low score)
            is_coasting = frame_data["score"] < 0.2
            color_override = (0, 165, 255) if is_coasting else None # Orange for coasting
            
            if is_coasting:
                cv2.putText(img, "COASTING (Low Score)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

            for i, ((p1, p2), original_color) in enumerate(lines):
                current_color = color_override if color_override else original_color

                if count == 0 and i == 0:
                    # Debug first line of first frame
                    cam_p1 = project_point_3d(p1, R, C)
                    uv1 = project_camera_to_pixel(cam_p1, K)
                    print(f"DEBUG Frame {count}: Point {p1} -> Cam {cam_p1.T} -> UV {uv1}")
                    
                if draw_line_3d(img, p1, p2, R, C, K, current_color):
                    lines_drawn += 1

            if count == 0:
                print(f"DEBUG: Drawn {lines_drawn} lines on frame {count}")

            # Draw Stumps
            # Center of width = 0
            stump_y_bat = -L_half
            stump_y_bow = L_half
            stump_height = 0.71 # 28 inches
            
            # Batting Stumps
            draw_line_3d(img, (stump_y_bat, 0, 0), (stump_y_bat, 0, stump_height), R, C, K, config.COLOR_STUMP, 4)
            # Bowling Stumps
            draw_line_3d(img, (stump_y_bow, 0, 0), (stump_y_bow, 0, stump_height), R, C, K, config.COLOR_STUMP, 4)
            
            cv2.putText(img, f"Frame: {count}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Save exact filename
        outfile = os.path.join(args.output, filename)
        cv2.imwrite(outfile, img)
        
        count += 1
        if count % 100 == 0:
            print(f"Visualized {count} frames...")
            
    if is_video:
        cap.release()
    print("Done visualization.")

if __name__ == "__main__":
    main()
