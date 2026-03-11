"""
Camera Tracker for Cricket Pitch — Soft Tripod Model

BroadTrack-style tracker with soft position constraint.
All 8 params are optimized per frame, but camera position is
strongly penalized for deviating from the previous frame's position.
This lets the position slowly correct while staying stable.

Uses scipy.optimize.least_squares (trust-region) with bounds.
"""

import numpy as np
import cv2
from scipy.optimize import least_squares

from camera import (
    project_point, project_points_batch, project_wireframe,
    focal_from_hfov, hfov_from_focal
)
import config


def point_to_line_dist(pt, line_pt1, line_pt2):
    """Distance from 2D point pt to the infinite line passing through line_pt1 and line_pt2."""
    l2 = np.sum((line_pt1 - line_pt2)**2)
    if l2 == 0:
        return np.linalg.norm(pt - line_pt1)
    num = np.abs((line_pt2[0] - line_pt1[0]) * (line_pt1[1] - pt[1]) - (line_pt1[0] - pt[0]) * (line_pt2[1] - line_pt1[1]))
    return num / np.sqrt(l2)


# ─────────────────────────────────────────────────────
# PITCH MODEL
# ─────────────────────────────────────────────────────

L = config.PITCH_LENGTH / 2.0  # 10.06m half-length
W = config.PITCH_WIDTH / 2.0   # 1.525m half-width
POP = 1.22  # Popping crease distance

# 8 keypoints as 3D points (Z=0 ground plane)
# 0-3: pitch corners (bowling crease × return crease)
# 4-7: popping crease × return crease
PITCH_CORNERS_3D = np.array([
    [-L,       -W, 0],  # 0: batting_left_corner
    [-L,        W, 0],  # 1: batting_right_corner
    [ L,        W, 0],  # 2: bowling_right_corner
    [ L,       -W, 0],  # 3: bowling_left_corner
    [-L + POP, -W, 0],  # 4: batting_left_pop
    [-L + POP,  W, 0],  # 5: batting_right_pop
    [ L - POP,  W, 0],  # 6: bowling_right_pop
    [ L - POP, -W, 0],  # 7: bowling_left_pop
], dtype=np.float64)

# Wireframe lines for visualization
PITCH_WIREFRAME = [
    # Boundary
    ([-L, -W, 0], [-L, W, 0]),   # Batting end
    ([L, -W, 0], [L, W, 0]),     # Bowling end
    ([-L, -W, 0], [L, -W, 0]),   # Left side
    ([-L, W, 0], [L, W, 0]),     # Right side
    # Creases
    ([-L + POP, -W, 0], [-L + POP, W, 0]),  # Batting popping crease
    ([L - POP, -W, 0], [L - POP, W, 0]),    # Bowling popping crease
]


# ─────────────────────────────────────────────────────
# COST FUNCTION
# ─────────────────────────────────────────────────────

def soft_tripod_residuals(params, keypoints_2d, keypoints_3d, principal_point,
                          prev_params=None, position_weight=20.0, focal_weight=0.05,
                          rotation_weight=10.0, k1_weight=0.1,
                          edge_points=None, use_lines=False,
                          line_weight=1.0, keypoint_weights=None):
    """
    Combined cost function for soft-tripod model.
    """
    residuals = []
    
    if keypoint_weights is None:
        keypoint_weights = np.ones(len(keypoints_2d))
    
    # 1. Keypoint reprojection
    for pt_2d, pt_3d, w in zip(keypoints_2d, keypoints_3d, keypoint_weights):
        projected = project_point(params, pt_3d, principal_point)
        if projected is not None:
            residuals.append((projected[0] - pt_2d[0]) * w)
            residuals.append((projected[1] - pt_2d[1]) * w)
        else:
            residuals.append(500.0 * w)
            residuals.append(500.0 * w)
    
    # 2. Temporal smoothness (if we have a previous frame)
    if prev_params is not None:
        # Rotation: mathematically penalized to prevent jitter
        for i in range(0, 3):
            residuals.append(rotation_weight * (params[i] - prev_params[i]))
            
        # Position: strongly penalized (camera is on a tripod!)
        for i in range(3, 6):
            residuals.append(position_weight * (params[i] - prev_params[i]))
        
        # Focal length: moderately penalized (zoom changes gradually)
        residuals.append(focal_weight * (params[6] - prev_params[6]))
        
        # k1 distortion: penalized to change slowly (lens doesn't change)
        residuals.append(k1_weight * (params[7] - prev_params[7]))
        
    # 3. Line residuals (Method A)
    if use_lines and edge_points is not None and len(edge_points) > 0:
        corners_2d, valid = project_points_batch(params, PITCH_CORNERS_3D, principal_point)
        if np.all(valid):
            c0, c1, c2, c3 = corners_2d[0], corners_2d[1], corners_2d[2], corners_2d[3]
            # Define 4 lines matching pitch outline
            lines = [(c0, c1), (c1, c2), (c2, c3), (c3, c0)]
            
            for pt in edge_points:
                pt_arr = np.array(pt)
                min_dist = float('inf')
                for l_p1, l_p2 in lines:
                    d = point_to_line_dist(pt_arr, l_p1, l_p2)
                    if d < min_dist:
                        min_dist = d
                residuals.append(min_dist * line_weight)
        else:
            # Huge penalty if corners project behind camera
            for _ in edge_points:
                residuals.append(500.0)
    
    return np.array(residuals)


# ─────────────────────────────────────────────────────
# CAMERA TRACKER
# ─────────────────────────────────────────────────────

class CameraTracker:
    """
    Soft-tripod camera tracker.
    
    All 8 params optimized per frame, but position strongly constrained
    to stay near previous frame (tripod doesn't move).
    Position CAN slowly drift to correct initialization errors.
    """
    
    HFOV_MIN_DEG = 5.0
    HFOV_MAX_DEG = 70.0
    
    def __init__(self, image_width=1920, image_height=1080,
                 tripod_position=None, tripod_radius=None):
        self.image_width = image_width
        self.image_height = image_height
        self.principal_point = np.array([image_width / 2.0, image_height / 2.0])
        
        self.params = None       # Full 8-param vector
        self.prev_params = None
        
        # Tripod position lock (from compute_tripod.py Pass 2)
        # When set, XYZ position is locked — only rotation + focal + k1 can change
        self.tripod_position = tripod_position  # (3,) array or None
        self.tripod_radius = tripod_radius       # float or None
        
        # Legacy: when reinitializing, remember the tripod's Z height
        self.known_tripod_z = None
        
        self.initialized = False
        
        # Focal bounds
        self.focal_min = focal_from_hfov(self.HFOV_MAX_DEG, image_width)
        self.focal_max = focal_from_hfov(self.HFOV_MIN_DEG, image_width)
    
    def _get_bounds(self):
        """Bounds for all 8 params."""
        
        if self.tripod_position is not None:
            # Pass 2: Lock XYZ tightly around the tripod pivot
            tp = self.tripod_position
            r = self.tripod_radius if self.tripod_radius else 0.05
            margin = max(r, 0.05)  # At least 5cm margin
            lower = [-np.inf, -np.inf, -np.inf,       # rotation: unbounded
                     tp[0] - margin, tp[1] - margin, tp[2] - margin,
                     self.focal_min,
                     -0.3]
            upper = [np.inf, np.inf, np.inf,
                     tp[0] + margin, tp[1] + margin, tp[2] + margin,
                     self.focal_max,
                     0.3]
        else:
            lower = [-np.inf, -np.inf, -np.inf,   # rotation: unbounded
                     -200.0, -200.0, 5.0,          # position: Z > 5m
                     self.focal_min,
                     -0.3]
            upper = [np.inf, np.inf, np.inf,
                     200.0, 200.0, 100.0,
                     self.focal_max,
                     0.3]
                 
            # If we know the tripod height, lock the Z bound tightly
            if self.known_tripod_z is not None:
                lower[5] = self.known_tripod_z - 1.0
                upper[5] = self.known_tripod_z + 1.0
            
        return (lower, upper)
    
    def reinit(self, keypoints_2d, keypoints_3d):
        """
        Initialize from YOLO keypoints using PnP.
        
        Uses a fixed reasonable hfov (25°) for PnP, then lets the
        optimizer refine all params including focal length.
        """
        if len(keypoints_2d) < 4:
            return False
        
        best_reproj = float('inf')
        best_params = None
        
        # Try a range of hfov values
        for hfov_deg in range(10, 65, 5):
            focal = focal_from_hfov(hfov_deg, self.image_width)
            K = np.array([
                [focal, 0, self.principal_point[0]],
                [0, focal, self.principal_point[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            
            success, rvec, tvec = cv2.solvePnP(
                keypoints_3d.astype(np.float64),
                keypoints_2d.astype(np.float64),
                K, np.zeros(4),
                flags=cv2.SOLVEPNP_IPPE
            )
            
            if not success:
                continue
            
            R, _ = cv2.Rodrigues(rvec)
            position = (-R.T @ tvec).flatten()
            
            # Broadcast cameras are elevated in the stands (typically 10-30m).
            # Reject solutions placing the camera below 5m — those are
            # degenerate PnP flips from near-planar geometry.
            if position[2] < 5.0:
                continue
            
            # Build initial 8-param vector (clipped to bounds)
            bounds = self._get_bounds()
            params0 = np.zeros(8)
            params0[0:3] = rvec.flatten()
            params0[3:6] = position
            
            # Force PnP initialization to snap back to the known tripod height BEFORE optimizing
            if self.known_tripod_z is not None:
                params0[5] = self.known_tripod_z
                
            params0[6] = focal
            params0[7] = 0.0
            
            # Clip to bounds to prevent least_squares ValueError crash
            params0 = np.clip(params0, bounds[0], bounds[1])
            
            # Quick optimization (no temporal constraint during init)
            try:
                result = least_squares(
                    soft_tripod_residuals,
                    params0,
                    args=(keypoints_2d, keypoints_3d, self.principal_point,
                          None, 0.0, 0.0),  # No temporal constraint
                    method='trf',
                    bounds=bounds,
                    max_nfev=300
                )
            except ValueError:
                continue
            
            # Score by reproj error
            reproj = self._mean_reproj_error(result.x, keypoints_2d, keypoints_3d)
            
            if reproj < best_reproj:
                best_reproj = reproj
                best_params = result.x.copy()
        
        if best_params is not None and best_reproj < 50:
            self.params = best_params
            self.prev_params = best_params.copy()
            self.known_tripod_z = best_params[5]  # Save the resolved tripod height
            self.initialized = True
            return True
        
        return False
    
    def update(self, keypoints_2d, keypoints_3d, 
               position_weight=20.0, focal_weight=0.05,
               rotation_weight=10.0,
               outlier_threshold=30.0,
               edge_points=None, use_lines=False, use_predictive_anchors=False):
        """
        Per-frame update with outlier rejection and optional constraints.
        """
        if not self.initialized or self.params is None:
            return None, None
            
        kp_2d = list(keypoints_2d)
        kp_3d = list(keypoints_3d)
        kp_weights = [1.0] * len(kp_2d)
        
        # Method B: Predictive Anchoring
        if use_predictive_anchors and self.prev_params is not None and len(kp_2d) < 4:
            for i, corner in enumerate(PITCH_CORNERS_3D):
                # Check if this corner is already in kp_3d
                found = any(np.allclose(corner, existing_3d) for existing_3d in kp_3d)
                if not found:
                    proj = project_point(self.prev_params, corner, self.principal_point)
                    if proj is not None:
                        kp_2d.append(proj)
                        kp_3d.append(corner)
                        kp_weights.append(0.2)  # Low weight so it's a weak anchor
                        
        kp_2d = np.array(kp_2d)
        kp_3d = np.array(kp_3d)
        kp_weights = np.array(kp_weights)
        
        if len(kp_2d) < 3:
            return 0.0, self.params.copy()
        
        params0 = self.params.copy()
        
        # --- Pass 1: fit with all keypoints ---
        result = least_squares(
            soft_tripod_residuals,
            params0,
            args=(kp_2d, kp_3d, self.principal_point,
                  self.prev_params, position_weight, focal_weight, rotation_weight,
                  0.1, edge_points, use_lines, 1.0, kp_weights),
            method='trf',
            bounds=self._get_bounds(),
            max_nfev=200
        )
        
        new_params = result.x
        
        # --- Outlier rejection ---
        per_point_errors = self._per_point_reproj_errors(new_params, kp_2d, kp_3d)
        inlier_mask = per_point_errors < outlier_threshold
        n_inliers = int(np.sum(inlier_mask))
        
        if n_inliers < len(kp_2d) and n_inliers >= 3:
            # Re-fit with inliers only
            kp_2d_clean = kp_2d[inlier_mask]
            kp_3d_clean = kp_3d[inlier_mask]
            kp_weights_clean = kp_weights[inlier_mask]
            
            result2 = least_squares(
                soft_tripod_residuals,
                new_params,  # warm start from pass 1
                args=(kp_2d_clean, kp_3d_clean, self.principal_point,
                      self.prev_params, position_weight, focal_weight, rotation_weight,
                      0.1, edge_points, use_lines, 1.0, kp_weights_clean),
                method='trf',
                bounds=self._get_bounds(),
                max_nfev=200
            )
            new_params = result2.x
            
        reproj = self._mean_reproj_error(new_params, np.array(keypoints_2d), np.array(keypoints_3d))
        
        if reproj > 200 or n_inliers < 3:
            return None, None
        
        # Accept
        self.prev_params = self.params.copy()
        self.params = new_params
        
        return reproj, new_params
    
    def get_wireframe_2d(self):
        """Project pitch wireframe using current camera params."""
        if self.params is None:
            return []
        return project_wireframe(self.params, PITCH_WIREFRAME, self.principal_point)
    
    def get_corners_2d(self):
        """Project pitch corners using current camera params."""
        if self.params is None:
            return None, None
        return project_points_batch(self.params, PITCH_CORNERS_3D, self.principal_point)
    
    def get_camera_info(self):
        """Get human-readable camera info."""
        if self.params is None:
            return {}
        return {
            "position": self.params[3:6].tolist(),
            "focal": float(self.params[6]),
            "hfov_deg": hfov_from_focal(self.params[6], self.image_width),
        }
    
    def _mean_reproj_error(self, params, keypoints_2d, keypoints_3d):
        """Mean reprojection error in pixels."""
        projected, valid = project_points_batch(params, keypoints_3d, self.principal_point)
        if not np.any(valid):
            return float('inf')
        errors = np.linalg.norm(projected[valid] - keypoints_2d[valid], axis=1)
        return float(np.mean(errors))
    
    def _per_point_reproj_errors(self, params, keypoints_2d, keypoints_3d):
        """Per-keypoint reprojection error in pixels."""
        n = len(keypoints_2d)
        errors = np.full(n, 999.0)
        for i in range(n):
            proj = project_point(params, keypoints_3d[i], self.principal_point)
            if proj is not None:
                errors[i] = np.linalg.norm(proj - keypoints_2d[i])
        return errors
