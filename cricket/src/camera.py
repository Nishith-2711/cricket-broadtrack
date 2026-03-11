"""
Camera Model for Cricket Pitch Tracking

Direct Python translation of BroadTrack's Camera.h.
Camera parameters: [ax, ay, az, px, py, pz, focal, k1]
  - ax, ay, az: angle-axis rotation vector
  - px, py, pz: camera position in world coordinates
  - focal: focal length in pixels
  - k1: radial distortion coefficient

Projection pipeline:
  world_point → translate → rotate → normalize → distort → pixel
"""

import numpy as np
import cv2
import math


def project_point(params, world_pt, principal_point):
    """
    Project a 3D world point to 2D pixel coordinates.
    
    Args:
        params: [ax, ay, az, px, py, pz, focal, k1] (8 params)
        world_pt: (3,) array — world coordinates
        principal_point: (2,) array — [cx, cy] pixel center
    
    Returns:
        (2,) array — pixel coordinates, or None if behind camera
    """
    angle_axis = params[0:3]
    position = params[3:6]
    focal = params[6]
    k1 = params[7]
    
    # Translate: world → camera-centered
    p = world_pt - position
    
    # Rotate: world → camera frame
    R, _ = cv2.Rodrigues(np.array(angle_axis, dtype=np.float64))
    cam_pt = R @ p
    
    # Check if point is in front of camera
    if cam_pt[2] <= 0:
        return None
    
    # Normalize (project onto z=1 plane)
    nx = cam_pt[0] / cam_pt[2]
    ny = cam_pt[1] / cam_pt[2]
    
    # Radial distortion
    r2 = nx * nx + ny * ny
    distortion = 1.0 + k1 * r2
    
    # Apply focal length → pixel coordinates
    px = focal * distortion * nx + principal_point[0]
    py = focal * distortion * ny + principal_point[1]
    
    return np.array([px, py])


def project_points_batch(params, world_pts, principal_point):
    """
    Project multiple 3D points to 2D.
    
    Returns:
        projected: (N, 2) array of pixel coords
        valid_mask: (N,) bool array
    """
    N = len(world_pts)
    projected = np.zeros((N, 2))
    valid_mask = np.zeros(N, dtype=bool)
    
    for i, wp in enumerate(world_pts):
        result = project_point(params, wp, principal_point)
        if result is not None:
            projected[i] = result
            valid_mask[i] = True
    
    return projected, valid_mask


def project_wireframe(params, pitch_lines, principal_point):
    """
    Project pitch wireframe (list of 3D line segments) to 2D.
    
    Args:
        params: camera parameters (8,)
        pitch_lines: list of (start_3d, end_3d) tuples
        principal_point: (2,) principal point
    
    Returns:
        list of (start_2d, end_2d) tuples for valid lines
    """
    lines_2d = []
    for start_3d, end_3d in pitch_lines:
        p1 = project_point(params, np.array(start_3d), principal_point)
        p2 = project_point(params, np.array(end_3d), principal_point)
        if p1 is not None and p2 is not None:
            lines_2d.append((p1, p2))
    return lines_2d


def get_ground_homography(params, principal_point):
    """
    Get the 3x3 homography that maps ground plane (Z=0) to image pixels.
    H such that: [u, v, 1]^T ~ H @ [X, Y, 1]^T
    """
    angle_axis = params[0:3]
    position = params[3:6]
    focal = params[6]
    
    R, _ = cv2.Rodrigues(np.array(angle_axis, dtype=np.float64))
    t = -R @ position
    
    # For Z=0 plane, homography is H = K @ [r1, r2, t]
    K = np.array([
        [focal, 0, principal_point[0]],
        [0, focal, principal_point[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    Rt_plane = np.column_stack([R[:, 0], R[:, 1], t])
    H = K @ Rt_plane
    
    return H


def params_from_pan_tilt_roll(pan, tilt, roll, position, focal, k1=0.0):
    """
    Convert human-readable camera params to the 8-param vector.
    
    Args:
        pan: pan angle in radians (rotation around Z axis)
        tilt: tilt angle in radians (rotation around X axis)
        roll: roll angle in radians
        position: (3,) camera position in world coords
        focal: focal length in pixels
        k1: radial distortion
    
    Returns:
        (8,) parameter vector
    """
    # Build rotation matrix from pan-tilt-roll (ZXY convention like BroadTrack)
    Rz = np.array([
        [math.cos(pan), -math.sin(pan), 0],
        [math.sin(pan),  math.cos(pan), 0],
        [0, 0, 1]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(tilt), -math.sin(tilt)],
        [0, math.sin(tilt),  math.cos(tilt)]
    ])
    Rroll = np.array([
        [math.cos(roll), -math.sin(roll), 0],
        [math.sin(roll),  math.cos(roll), 0],
        [0, 0, 1]
    ])
    
    R = Rroll @ Rx @ Rz
    angle_axis, _ = cv2.Rodrigues(R)
    
    params = np.zeros(8)
    params[0:3] = angle_axis.flatten()
    params[3:6] = position
    params[6] = focal
    params[7] = k1
    
    return params


def focal_from_hfov(hfov_deg, image_width):
    """Convert horizontal FOV to focal length in pixels."""
    return (image_width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def hfov_from_focal(focal, image_width):
    """Convert focal length to horizontal FOV in degrees."""
    return math.degrees(2.0 * math.atan((image_width / 2.0) / focal))
