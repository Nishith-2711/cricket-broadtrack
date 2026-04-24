import os
import sys
import json
import argparse
import cv2
import numpy as np
import math

# Use the existing camera model utilities from inside your Python pipeline!
from broadtrack.camera import (
    project_wireframe,
    focal_from_hfov
)
from broadtrack.camera_tracker_sam import PITCH_WIREFRAME


def rotation_matrix_pan_tilt_roll_cpp(pan_rad, tilt_rad, roll_rad):
    """
    Match cricket C++ Camera::getOrientationFromPanTiltRoll exactly:
    return Rp * Rt * Rr  (then world→camera in storage is this matrix's transpose).
    """
    cp, sp = math.cos(pan_rad), math.sin(pan_rad)
    ct, st = math.cos(tilt_rad), math.sin(tilt_rad)
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    Rt = np.array(
        [[1.0, 0.0, 0.0], [0.0, ct, -st], [0.0, st, ct]], dtype=np.float64
    )
    Rp = np.array(
        [[cp, -sp, 0.0], [sp, cp, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    Rr = np.array(
        [[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    return Rp @ Rt @ Rr


def main():
    parser = argparse.ArgumentParser(description="Visualize BroadTrack Output onto Video")
    parser.add_argument("-j", "--json", default="bt_output.json", help="Path to tracked JSON")
    parser.add_argument("-f", "--frames", required=True, help="Directory containing original sequence of frames")
    parser.add_argument("-o", "--output", default="tracked_video.mp4", help="Output MP4 file path")
    args = parser.parse_args()

    print(f"Loading tracking data from {args.json}...")
    with open(args.json, 'r') as f:
        data = json.load(f)

    # Filter out frames that failed to track (score <= 0 or no rvec)
    valid_frames = {}
    for k, v in data.items():
        if "cp" not in v:
            continue
        cp = v["cp"]
        # Python tracker format: has "rvec" directly
        if "rvec" in cp:
            valid_frames[k] = v
        # C++ tracker format: pan/tilt/roll (preferred) or axisAngle only
        elif "panDegrees" in cp or "axisAngleX" in cp:
            valid_frames[k] = v
    
    if not valid_frames:
        print("No valid tracked frames found in JSON!")
        return

    # Sort frames to guarantee sequential video rendering
    sorted_paths = sorted(valid_frames.keys())
    
    # Find the raw images
    first_frame_path = sorted_paths[0]
    first_frame_basename = os.path.basename(first_frame_path)
    local_img_path = os.path.join(args.frames, first_frame_basename)
    
    first_img = cv2.imread(local_img_path)
    if first_img is None:
        print(f"Failed to read raw image frame at {local_img_path}!")
        return

    h, w = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(args.output, fourcc, 30.0, (w, h))

    principal_point = np.array([w / 2.0, h / 2.0])

    print(f"Rendering {len(sorted_paths)} frames to {args.output}...")
    
    for idx, path_key in enumerate(sorted_paths):
        frame_data = data[path_key]
        cp = frame_data["cp"]
        
        basename = os.path.basename(path_key)
        local_img_path = os.path.join(args.frames, basename)
        img = cv2.imread(local_img_path)
        if img is None:
            continue

        # Check which format we have
        if "rvec" in cp:
            # Python tracker format — rvec is the angle-axis directly
            rvec = np.array(cp["rvec"], dtype=np.float64)
            position = np.array([cp["positionXMeters"], cp["positionYMeters"], cp["positionZMeters"]])
            focal_pixels = cp["focal"]
            coeffs = cp.get("normalizedRadialDistortionCoefficients") or [0.0]
            dist = float(coeffs[0]) if len(coeffs) > 0 else 0.0

            params = np.zeros(8)
            params[0:3] = rvec
            params[3:6] = position
            params[6] = focal_pixels
            params[7] = dist

        elif "panDegrees" in cp:
            # C++ setPanTiltRoll: _rotationMatrix = getOrientationFromPanTiltRoll(...).transpose()
            # i.e. R_w2c = (Rp @ Rt @ Rr).T — NOT the old hand-derived r11..r33 matrix (that was wrong).
            hfov = cp["horizontalFieldOfViewDegrees"]
            pan_rad = math.radians(cp["panDegrees"])
            tilt_rad = math.radians(cp["tiltDegrees"])
            roll_rad = math.radians(cp["rollDegrees"])
            position = np.array(
                [
                    cp["positionXMeters"],
                    cp["positionYMeters"],
                    cp["positionZMeters"],
                ],
                dtype=np.float64,
            )
            coeffs = cp.get("normalizedRadialDistortionCoefficients") or [0.0]
            dist = float(coeffs[0]) if len(coeffs) > 0 else 0.0

            focal_pixels = focal_from_hfov(hfov, w)
            m = rotation_matrix_pan_tilt_roll_cpp(pan_rad, tilt_rad, roll_rad)
            r_w2c = m.T
            angle_axis, _ = cv2.Rodrigues(r_w2c)

            params = np.zeros(8)
            params[0:3] = angle_axis.flatten()
            params[3:6] = position
            params[6] = focal_pixels
            params[7] = dist

        elif "axisAngleX" in cp:
            # getOrientation() = _rotationMatrix.T = M with M = Rp@Rt@Rr; JSON axisAngle is Ceres
            # angle-axis for M. Python needs R_w2c = _rotationMatrix = M.T = Rodrigues(aa).T
            aa = np.array(
                [cp["axisAngleX"], cp["axisAngleY"], cp["axisAngleZ"]],
                dtype=np.float64,
            )
            r_m, _ = cv2.Rodrigues(aa)
            r_w2c = r_m.T
            rvec, _ = cv2.Rodrigues(r_w2c)

            position = np.array(
                [cp["positionXMeters"], cp["positionYMeters"], cp["positionZMeters"]],
                dtype=np.float64,
            )
            hfov = cp["horizontalFieldOfViewDegrees"]
            focal_pixels = focal_from_hfov(hfov, w)
            coeffs = cp.get("normalizedRadialDistortionCoefficients") or [0.0]
            dist = float(coeffs[0]) if len(coeffs) > 0 else 0.0

            params = np.zeros(8)
            params[0:3] = rvec.flatten()
            params[3:6] = position
            params[6] = focal_pixels
            params[7] = dist
        else:
            continue

        # Project 3D Pitch Wireframe into 2D Pixels!
        projected_lines = project_wireframe(params, PITCH_WIREFRAME, principal_point)

        # Draw lines onto the frame
        for start_2d, end_2d in projected_lines:
            try:
                x1, y1 = float(start_2d[0]), float(start_2d[1])
                x2, y2 = float(end_2d[0]), float(end_2d[1])
                if math.isfinite(x1) and math.isfinite(y1) and math.isfinite(x2) and math.isfinite(y2):
                    pt1 = (int(round(x1)), int(round(y1)))
                    pt2 = (int(round(x2)), int(round(y2)))
                    cv2.line(img, pt1, pt2, (0, 0, 255), 2)  # Red Lines
            except:
                pass

        out_video.write(img)
        
        if (idx+1) % 50 == 0:
            print(f"   Rendered {idx+1}/{len(sorted_paths)} frames")

    out_video.release()
    print(f"\n[SUCCESS] Final visualization video written to {args.output}")

if __name__ == "__main__":
    main()