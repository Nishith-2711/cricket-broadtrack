"""
Camera Tracker for Cricket Pitch — SAM3 Lines-Primary Mode

Based on camera_tracker.py (Soft Tripod Model) with 3 key changes:
  1. soft_tripod_residuals() handles empty keypoints_2d (lines provide all signal)
  2. update() gate allows lines-only tracking (remove len(kp_2d) < 3 hard gate)
  3. Dynamic line_weight in update() — edges dominate when keypoints are scarce
"""

import numpy as np
import cv2
import math
from scipy.optimize import least_squares

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from camera import (
    project_point, project_points_batch, project_wireframe,
    focal_from_hfov, hfov_from_focal
)
import config


def point_to_line_dist(pt, line_a, line_b):
    """Distance from 2D point pt to the INFINITE line passing through line_a and line_b.
    
    This replaces the finite segment distance to prevent Focal Collapse. 
    By allowing edge points to slide infinitely along the line, we prevent a false 
    horizontal mask cutoff (e.g. from a player) from artificially squishing 
    the pitch length and twisting the camera focal length.
    """
    ab = line_b - line_a
    l2 = np.dot(ab, ab)
    if l2 == 0:
        return np.linalg.norm(pt - line_a)
    # Project pt onto the infinite line (no clipping)
    t = np.dot(pt - line_a, ab) / l2
    closest = line_a + t * ab
    return np.linalg.norm(pt - closest)


def get_visible_corner_count(params, pitch_corners_3d, principal_point, img_w, img_h):
    """Count how many pitch corners are visible (in frame bounds and in front of camera).
    
    Returns:
        visible_count: number of corners with valid projection in frame
        all_visible: bool, True if all corners are visible
    """
    corners_2d, valid = project_points_batch(params, pitch_corners_3d, principal_point)
    if not np.any(valid):
        return 0, False
    
    visible = 0
    for i, (pt, is_valid) in enumerate(zip(corners_2d, valid)):
        if not is_valid:
            continue
        # Check if point is within frame bounds (not at edges)
        if pt[0] > 20 and pt[0] < (img_w - 20) and pt[1] > 20 and pt[1] < (img_h - 20):
            visible += 1
    
    return visible, visible >= len(pitch_corners_3d)


def _clip_segment_to_frame(p1, p2, img_w, img_h, margin=0):
    """Clip a 2D line segment to the frame boundaries using Cohen-Sutherland.
    
    Returns (clipped_p1, clipped_p2) or None if entirely outside.
    This ensures only the VISIBLE portion of a projected pitch edge
    participates in the cost function — matching soccer BroadTrack's behavior.
    """
    INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8
    xmin, xmax = margin, img_w - margin
    ymin, ymax = margin, img_h - margin
    
    def outcode(x, y):
        code = INSIDE
        if x < xmin: code |= LEFT
        elif x > xmax: code |= RIGHT
        if y < ymin: code |= TOP
        elif y > ymax: code |= BOTTOM
        return code
    
    x0, y0 = float(p1[0]), float(p1[1])
    x1, y1 = float(p2[0]), float(p2[1])
    
    code0 = outcode(x0, y0)
    code1 = outcode(x1, y1)
    
    for _ in range(20):  # max iterations
        if not (code0 | code1):
            # Both inside
            return (np.array([x0, y0]), np.array([x1, y1]))
        if code0 & code1:
            # Both outside on same side
            return None
        
        code_out = code0 if code0 else code1
        if code_out & TOP:
            x = x0 + (x1 - x0) * (ymin - y0) / (y1 - y0) if y1 != y0 else x0
            y = ymin
        elif code_out & BOTTOM:
            x = x0 + (x1 - x0) * (ymax - y0) / (y1 - y0) if y1 != y0 else x0
            y = ymax
        elif code_out & RIGHT:
            y = y0 + (y1 - y0) * (xmax - x0) / (x1 - x0) if x1 != x0 else y0
            x = xmax
        elif code_out & LEFT:
            y = y0 + (y1 - y0) * (xmin - x0) / (x1 - x0) if x1 != x0 else y0
            x = xmin
        
        if code_out == code0:
            x0, y0 = x, y
            code0 = outcode(x0, y0)
        else:
            x1, y1 = x, y
            code1 = outcode(x1, y1)
    
    return None

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
# DIRT STRIP MODEL (what SAM actually segments)
# ─────────────────────────────────────────────────────
# SAM3 segments the entire worn brown dirt strip, which is LARGER than
# the official 20.12m × 3.05m playing pitch. It extends beyond the
# bowling creases (popping crease approach + bowler's run-up wear) and
# is typically slightly wider than the return creases.
#
# These values are empirical — tune to match your SAM mask if needed.
# Starting estimate: ~24m × ~3.7m (the "visible brown rectangle").
L_DIRT = 12.0   # half-length → 24.0m total dirt strip length
W_DIRT = 1.85   # half-width  → 3.70m  total dirt strip width

DIRT_STRIP_CORNERS_3D = np.array([
    [-L_DIRT, -W_DIRT, 0],  # 0: batting-end, left
    [-L_DIRT,  W_DIRT, 0],  # 1: batting-end, right
    [ L_DIRT,  W_DIRT, 0],  # 2: bowling-end, right
    [ L_DIRT, -W_DIRT, 0],  # 3: bowling-end, left
], dtype=np.float64)

DIRT_STRIP_WIREFRAME = [
    ([-L_DIRT, -W_DIRT, 0], [-L_DIRT,  W_DIRT, 0]),  # Batting end
    ([ L_DIRT, -W_DIRT, 0], [ L_DIRT,  W_DIRT, 0]),  # Bowling end
    ([-L_DIRT, -W_DIRT, 0], [ L_DIRT, -W_DIRT, 0]),  # Left side
    ([-L_DIRT,  W_DIRT, 0], [ L_DIRT,  W_DIRT, 0]),  # Right side
]


# ─────────────────────────────────────────────────────
# COST FUNCTION (CHANGE 1: handles empty keypoints_2d)
# ─────────────────────────────────────────────────────

def soft_tripod_residuals(params, keypoints_2d, keypoints_3d, principal_point,
                          prev_params=None, position_weight=20.0, focal_weight=0.05,
                          rotation_weight=10.0, k1_weight=0.1,
                          edge_points=None, use_lines=False,
                          line_weight=1.0, keypoint_weights=None,
                          pitch_corners_3d=None, partial_visibility=False,
                          line_corners_3d=None):
    """
    Combined cost function for soft-tripod model.

    CHANGE 1: Handles empty keypoints_2d — when no keypoints are available,
    line residuals (edge_points) provide all geometric signal.

    Option B: `line_corners_3d` decouples the geometry used for line/mask
    residuals from the keypoint geometry. SAM segments the full dirt strip
    (~24m × 3.7m), which is larger than the official playing pitch
    (20.12m × 3.05m). Passing the larger dirt-strip rectangle here stops
    the line cost from fighting the keypoint cost and collapsing the
    projected rectangle inward.
    """
    residuals = []
    
    # ── CHANGE 1: Guard against empty keypoints ──
    n_kp = len(keypoints_2d) if keypoints_2d is not None else 0
    
    if keypoint_weights is None and n_kp > 0:
        keypoint_weights = np.ones(n_kp)
    
    # 1. Keypoint reprojection (only if we have keypoints)
    if n_kp > 0:
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
        
    # 3. Line residuals — only VISIBLE portions of projected pitch edges
    if use_lines and edge_points is not None and len(edge_points) > 0:
        # Option B: prefer the explicit line_corners_3d (dirt strip) when
        # provided; fall back to the keypoint pitch corners for legacy callers.
        if line_corners_3d is not None:
            line_corners = line_corners_3d
        elif pitch_corners_3d is not None:
            line_corners = pitch_corners_3d
        else:
            line_corners = PITCH_CORNERS_3D
        corners_2d, valid = project_points_batch(params, line_corners, principal_point)
        if np.all(valid):
            c0, c1, c2, c3 = corners_2d[0], corners_2d[1], corners_2d[2], corners_2d[3]
            # Define 4 segments matching pitch outline
            raw_lines = [(c0, c1), (c1, c2), (c2, c3), (c3, c0)]
            
            # Clip each projected segment to the frame boundary
            # Only the visible portion participates (like soccer BroadTrack)
            img_w = principal_point[0] * 2  # approximate frame width
            img_h = principal_point[1] * 2  # approximate frame height
            clipped_lines = []
            for p1, p2 in raw_lines:
                clipped = _clip_segment_to_frame(p1, p2, img_w, img_h)
                if clipped is not None:
                    clipped_lines.append(clipped)
            
            if len(clipped_lines) > 0:
                for pt in edge_points:
                    pt_arr = np.array(pt, dtype=np.float64)
                    min_dist = float('inf')
                    for l_p1, l_p2 in clipped_lines:
                        d = point_to_line_dist(pt_arr, l_p1, l_p2)
                        if d < min_dist:
                            min_dist = d
                    residuals.append(min_dist * line_weight)
            else:
                # Corners project in front of camera but every outline segment lies
                # outside the clip window. Must still emit one residual per edge
                # point — otherwise vector length changes across iterations and
                # scipy.least_squares raises broadcast errors.
                for _ in edge_points:
                    residuals.append(700.0 if partial_visibility else 500.0)
        else:
            # Huge penalty if corners project behind camera
            for _ in edge_points:
                residuals.append(700.0 if partial_visibility else 500.0)
    
    return np.array(residuals)


# ─────────────────────────────────────────────────────
# CAMERA TRACKER (CHANGES 2 & 3)
# ─────────────────────────────────────────────────────

class CameraTracker:
    """
    Soft-tripod camera tracker — SAM3 lines-primary variant.
    
    CHANGE 2: update() allows lines-only tracking (no keypoint minimum)
    CHANGE 3: Dynamic line_weight — edges dominate when keypoints are scarce
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
        self.tripod_position = tripod_position
        self.tripod_radius = tripod_radius
        
        # Legacy: when reinitializing, remember the tripod's Z height
        self.known_tripod_z = None
        
        self.initialized = False
        self.yaw_flip = False  # If True, negate X/Y world coords (batting/bowling swap)
        
        # Focal bounds
        self.focal_min = focal_from_hfov(self.HFOV_MAX_DEG, image_width)
        self.focal_max = focal_from_hfov(self.HFOV_MIN_DEG, image_width)
    
    def _get_bounds(self):
        """Bounds for all 8 params."""
        
        if self.tripod_position is not None:
            # Pass 2: Lock XYZ tightly around the tripod pivot
            tp = self.tripod_position
            r = self.tripod_radius if self.tripod_radius else 0.05
            # Use radius-adaptive lock: keep it tight, but avoid over-constraining
            # when low-pan fallback underestimates radius.
            margin = min(max(0.25, 0.50 * r), 0.80)
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
            # FIX: Tighten from \pm 1.0m to \pm 0.1m to prevent depth/focal ambiguity when tracking lines-only
            if self.known_tripod_z is not None:
                lower[5] = self.known_tripod_z - 0.1
                upper[5] = self.known_tripod_z + 0.1
            
        return (lower, upper)
    
    def reinit(self, keypoints_2d, keypoints_3d):
        """
        Initialize from YOLO keypoints using PnP.
        
        Tries both Standard and 180-degree Flipped orientations to solve
        batting/bowling end ambiguity.
        """
        if len(keypoints_2d) < 4:
            return False
        
        best_reproj = float('inf')
        best_params = None
        
        # Try both Standard and Flipped (180 deg rotation) 3D mappings
        standard_3d = keypoints_3d.astype(np.float64)
        flipped_3d = standard_3d.copy()
        flipped_3d[:, 0] *= -1
        flipped_3d[:, 1] *= -1

        variant_errors = []
        for i, kp_3d_variant in enumerate([standard_3d, flipped_3d]):
            variant_name = "STANDARD" if i == 0 else "FLIPPED"
            best_variant_reproj = float('inf')
            best_variant_params = None
            
            for hfov_deg in range(10, 65, 5):
                focal = focal_from_hfov(hfov_deg, self.image_width)
                K = np.array([
                    [focal, 0, self.principal_point[0]],
                    [0, focal, self.principal_point[1]],
                    [0, 0, 1]
                ], dtype=np.float64)
                
                success, rvec, tvec = cv2.solvePnP(
                    kp_3d_variant,
                    keypoints_2d.astype(np.float64),
                    K, np.zeros(4),
                    flags=cv2.SOLVEPNP_IPPE
                )
                
                if not success:
                    continue
                
                R, _ = cv2.Rodrigues(rvec)
                position = (-R.T @ tvec).flatten()
                
                # Reject solutions placing camera below 5m (degenerate PnP flips)
                if position[2] < 5.0:
                    continue
                
                # Build initial 8-param vector
                bounds = self._get_bounds()
                params0 = np.zeros(8)
                params0[0:3] = rvec.flatten()
                params0[3:6] = position
                if self.known_tripod_z is not None:
                    params0[5] = self.known_tripod_z
                params0[6] = focal
                params0[7] = 0.0
                
                params0 = np.clip(params0, bounds[0], bounds[1])
                
                try:
                    result = least_squares(
                        soft_tripod_residuals,
                        params0,
                        args=(keypoints_2d, kp_3d_variant, self.principal_point,
                              None, 0.0, 0.0, 0.0, 0.1,
                              None, False, 1.0, None, 
                              kp_3d_variant),
                        method='trf',
                        loss='soft_l1',
                        f_scale=25.0,
                        bounds=bounds,
                        max_nfev=300
                    )
                    reproj = self._mean_reproj_error(result.x, keypoints_2d, kp_3d_variant)
                    if reproj < best_variant_reproj:
                        best_variant_reproj = reproj
                        best_variant_params = result.x.copy()
                except ValueError:
                    continue
            
            variant_errors.append(best_variant_reproj)
            
            # Apply 0.5px bias to Standard mapping to avoid flickering in symmetric cases
            score = best_variant_reproj
            if i == 0:  # STANDARD
                score -= 0.5
                
            if score < best_reproj:
                best_reproj = score
                final_error = best_variant_reproj
                best_params = best_variant_params
                self.yaw_flip = (i == 1)
        
        if best_params is not None and best_reproj < 50:
            self.params = best_params
            self.prev_params = best_params.copy()
            self.known_tripod_z = best_params[5]
            self.initialized = True
            print(f"  [REINIT] orient={('FLIPPED' if self.yaw_flip else 'STANDARD')}, error={final_error:.2f}px (STD={variant_errors[0]:.1f}, FLP={variant_errors[1]:.1f})")
            return True
        
        return False
    
    def update(self, keypoints_2d, keypoints_3d, 
               position_weight=20.0, focal_weight=0.05,
               rotation_weight=10.0,
               outlier_threshold=30.0,
               edge_points=None, use_lines=False, use_predictive_anchors=False,
               line_weight=1.0, kp_weight_scale=1.0,
               partial_visibility=False):
        """
        Per-frame update with outlier rejection and optional constraints.
        
        CHANGE 2: Allows lines-only when edge_points provide enough signal.
        CHANGE 3: Accepts dynamic line_weight and kp_weight_scale from caller.
        """
        if not self.initialized or self.params is None:
            return None, None, 0

        kp_2d = list(keypoints_2d) if keypoints_2d is not None and len(keypoints_2d) > 0 else []
        kp_3d = list(keypoints_3d) if keypoints_3d is not None and len(keypoints_3d) > 0 else []
        
        # Apply detected orientation flip
        if self.yaw_flip and len(kp_3d) > 0:
            kp_3d_flipped = []
            for pt in kp_3d:
                kp_3d_flipped.append(np.array([-pt[0], -pt[1], pt[2]]))
            kp_3d = kp_3d_flipped
        
        # Apply keypoint weight scaling — when SAM edges are present,
        # keypoints are downweighted so edges drive the solution
        kp_weights = [kp_weight_scale] * len(kp_2d)
        
        # Method B: Predictive Anchoring
        if use_predictive_anchors and self.prev_params is not None and len(kp_2d) < 4:
            # Use correctly oriented corners for anchoring
            anchors_3d = PITCH_CORNERS_3D
            if self.yaw_flip:
                anchors_3d = anchors_3d.copy()
                anchors_3d[:, 0:2] *= -1

            for i, corner in enumerate(anchors_3d):
                found = any(np.allclose(corner, existing_3d) for existing_3d in kp_3d)
                if not found:
                    proj = project_point(self.prev_params, corner, self.principal_point)
                    if proj is not None:
                        kp_2d.append(proj)
                        kp_3d.append(corner)
                        kp_weights.append(0.2)
                        
        kp_2d = np.array(kp_2d) if len(kp_2d) > 0 else np.empty((0, 2), dtype=np.float64)
        kp_3d = np.array(kp_3d) if len(kp_3d) > 0 else np.empty((0, 3), dtype=np.float64)
        kp_weights = np.array(kp_weights) if len(kp_weights) > 0 else np.array([])
        
        # --- Phase 3: Predictive Validation (Anti-Hallucination) ---
        if self.prev_params is not None and len(kp_2d) > 0:
            predicted_2d, valid_proj = project_points_batch(
                self.prev_params, kp_3d, self.principal_point
            )
            if np.all(valid_proj):
                pred_threshold = 150.0 
                
                valid_mask = []
                for i in range(len(kp_2d)):
                    dist = np.linalg.norm(kp_2d[i] - predicted_2d[i])
                    valid_mask.append(dist < pred_threshold)
                
                valid_mask = np.array(valid_mask)
                kp_2d = kp_2d[valid_mask]
                kp_3d = kp_3d[valid_mask]
                kp_weights = kp_weights[valid_mask]

        # ── CHANGE 2: Allow lines-only tracking ──
        has_enough_lines = use_lines and edge_points is not None and len(edge_points) >= 6
        if len(kp_2d) < 1 and not has_enough_lines:
            # Only fail if BOTH keypoints AND lines are absent
            return None, None, 0
        
        params0 = self.params.copy()
        
        # Get correctly oriented pitch corners for Method A
        anchors_3d = PITCH_CORNERS_3D
        if self.yaw_flip:
            anchors_3d = anchors_3d.copy()
            anchors_3d[:, 0:2] *= -1

        # Option B: dirt-strip rectangle used ONLY for line/mask residuals.
        # Must be yaw-flipped to match the current batting/bowling orientation.
        dirt_corners_3d = DIRT_STRIP_CORNERS_3D
        if self.yaw_flip:
            dirt_corners_3d = dirt_corners_3d.copy()
            dirt_corners_3d[:, 0:2] *= -1

        # ── CHANGE 3: Use dynamic line_weight passed from caller ──
        result = least_squares(
            soft_tripod_residuals,
            params0,
            args=(kp_2d, kp_3d, self.principal_point,
                  self.prev_params, position_weight, focal_weight, rotation_weight,
                0.1, edge_points, use_lines, line_weight, kp_weights,
                anchors_3d, partial_visibility, dirt_corners_3d),
            method='trf',
            loss='soft_l1',
            f_scale=25.0,
            bounds=self._get_bounds(),
            max_nfev=200
        )
        
        new_params = result.x

        # --- Outlier rejection (only if we have keypoints to reject) ---
        # Track the inlier subset so the reject check uses a meaningful mean.
        kp_2d_eval = kp_2d
        kp_3d_eval = kp_3d
        if len(kp_2d) >= 3:
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
                          0.1, edge_points, use_lines, line_weight, kp_weights_clean,
                          anchors_3d, partial_visibility, dirt_corners_3d),
                    method='trf',
                    loss='soft_l1',
                    f_scale=25.0,
                    bounds=self._get_bounds(),
                    max_nfev=200
                )
                new_params = result2.x
                kp_2d_eval = kp_2d_clean
                kp_3d_eval = kp_3d_clean

        # Compute reprojection error on the inlier subset (what was actually
        # fit), plus the full-set error as a sanity check for gross drift.
        if len(kp_2d_eval) > 0:
            reproj = self._mean_reproj_error(new_params, kp_2d_eval, kp_3d_eval)
            full_reproj = self._mean_reproj_error(new_params, kp_2d, kp_3d)
        else:
            # Lines-only mode: use the optimizer cost as quality metric
            reproj = float(result.cost) / max(len(edge_points), 1)
            full_reproj = reproj

        # Reject for extreme errors. Two thresholds when keypoints are present:
        #   - inlier mean must be tight (< 15 px) — the fit itself must be good
        #   - full-set mean must be reasonable (< 40 px) — rules out gross drift
        #     that an outlier refit managed to mask by excluding most points
        n_kp_used = len(kp_2d)
        if n_kp_used >= 3:
            if reproj > 15.0 or full_reproj > 40.0:
                return None, None, 0
        else:
            max_acceptable = 800 if partial_visibility else 500
            if reproj > max_acceptable:
                return None, None, 0

        # Accept
        self.prev_params = self.params.copy()
        self.params = new_params

        return reproj, new_params, n_kp_used
    
    def get_wireframe_2d(self):
        """Project pitch wireframe using current camera params."""
        if self.params is None:
            return []
        
        wf = PITCH_WIREFRAME
        if self.yaw_flip:
            flipped_wf = []
            for line in PITCH_WIREFRAME:
                flipped_line = []
                for pt in line:
                    flipped_line.append(np.array([-pt[0], -pt[1], pt[2]]))
                flipped_wf.append(flipped_line)
            wf = flipped_wf
            
        return project_wireframe(self.params, wf, self.principal_point)
    
    def get_corners_2d(self):
        """Project pitch corners using current camera params."""
        if self.params is None:
            return None, None

        corners = PITCH_CORNERS_3D
        if self.yaw_flip:
            corners = corners.copy()
            corners[:, 0:2] *= -1

        return project_points_batch(self.params, corners, self.principal_point)

    def get_dirt_strip_2d(self):
        """Project the dirt-strip wireframe (larger rectangle SAM segments).

        Use this alongside get_wireframe_2d() to visualize what the line-
        residual geometry actually looks like vs. the official playing pitch.
        """
        if self.params is None:
            return []

        wf = DIRT_STRIP_WIREFRAME
        if self.yaw_flip:
            flipped_wf = []
            for line in DIRT_STRIP_WIREFRAME:
                flipped_line = []
                for pt in line:
                    flipped_line.append(np.array([-pt[0], -pt[1], pt[2]]))
                flipped_wf.append(flipped_line)
            wf = flipped_wf

        return project_wireframe(self.params, wf, self.principal_point)

    def get_dirt_strip_corners_2d(self):
        """Project dirt-strip corners — useful for scale/drift diagnostics."""
        if self.params is None:
            return None, None

        corners = DIRT_STRIP_CORNERS_3D
        if self.yaw_flip:
            corners = corners.copy()
            corners[:, 0:2] *= -1

        return project_points_batch(self.params, corners, self.principal_point)
    
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
        if len(keypoints_2d) == 0:
            return 0.0
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
