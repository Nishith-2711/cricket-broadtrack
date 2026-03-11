"""
Tripod Parameter Estimation for Cricket Camera Tracking

Adapted from BroadTrack's compute_tripod.py for our JSON format
(uses rvec instead of pan/tilt/roll).

Algorithm:
  1. Read Pass 1 JSON → extract camera positions + optical axes
  2. Compute pairwise intersections of optical axes
  3. Robust mean via MinCovDet (rejects outlier intersections)
  4. Fit a sphere (center + radius) to the camera positions
  5. Output tripod.json with {"sphere": {"center": [X,Y,Z], "radius": R}}

Usage:
  python compute_tripod.py -i bt_output_8_pts.json -o tripod.json
"""

import json
import argparse
import numpy as np
import cv2
from scipy.optimize import least_squares

try:
    from sklearn.covariance import MinCovDet
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from camera import focal_from_hfov


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def extract_camera_rays(json_path, min_score=0.6):
    """
    Extract camera positions and optical axes from a Pass 1 JSON file.

    Returns:
        positions: list of (3,) arrays — camera XYZ in world coords
        optical_axes: list of (3,) arrays — unit look-at vectors
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    positions = []
    optical_axes = []

    for key, entry in data.items():
        score = entry.get("score", 0)
        if score < min_score:
            continue

        cp = entry.get("cp", {})
        if "rvec" not in cp:
            continue

        rvec = np.array(cp["rvec"], dtype=np.float64)
        position = np.array([
            cp["positionXMeters"],
            cp["positionYMeters"],
            cp["positionZMeters"]
        ], dtype=np.float64)

        # Convert angle-axis → rotation matrix
        R, _ = cv2.Rodrigues(rvec)

        # The optical axis (look-at vector) is the 3rd row of R
        # (the camera looks along its local Z axis, which in world coords is R[2,:])
        look_at = R[2, :]

        positions.append(position)
        optical_axes.append(look_at)

    return positions, optical_axes


def compute_intersection_points(positions, axes, max_cos_similarity=0.995):
    """
    Compute pairwise intersections of optical axis rays.

    For each pair of rays (origin_i, dir_i) and (origin_j, dir_j),
    find the closest point between the two rays.

    Args:
        positions: list of (3,) origin points
        axes: list of (3,) direction vectors
        max_cos_similarity: skip nearly-parallel rays

    Returns:
        (N, 3) array of intersection midpoints
    """
    points = []
    n = len(positions)

    for i in range(n):
        for j in range(i + 1, n):
            d1 = axes[i]
            d2 = axes[j]

            # Skip nearly-parallel optical axes
            cos_theta = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2))
            if abs(cos_theta) > max_cos_similarity:
                continue

            # Find closest point between two rays via least squares
            # ray1: p1 + t * d1,  ray2: p2 + s * d2
            # Solve [d1, -d2] @ [t, s]^T = p2 - p1
            A = np.column_stack([d1, -d2])
            b = positions[j] - positions[i]

            try:
                result = np.linalg.lstsq(A, b, rcond=None)
                t, s = result[0]
                point1 = positions[i] + t * d1
                point2 = positions[j] + s * d2
                intersection = (point1 + point2) / 2.0
                points.append(intersection)
            except np.linalg.LinAlgError:
                continue

    return np.array(points) if points else np.array([]).reshape(0, 3)


def fit_tripod_sphere(positions, axes, initial_center):
    """
    Fit a sphere to the camera positions.

    The tripod model says all camera positions lie on a sphere
    centered at the tripod pivot. The camera pan/tilt moves the
    eye point along this sphere surface.

    Args:
        positions: list of (3,) camera positions
        axes: list of (3,) optical axes
        initial_center: (3,) initial guess for sphere center

    Returns:
        center: (3,) sphere center
        radius: float sphere radius
    """
    points = np.array(positions)
    directions = np.array(axes)

    def residuals(params, pts, dirs):
        c = params[:3]
        r = params[3]
        res = []
        for i in range(len(pts)):
            p = pts[i]
            d = dirs[i]
            # Find the point on the ray closest to center c
            lambda_i = np.dot(c - p, d) / np.dot(d, d)
            t = p + lambda_i * d
            # The distance from t to c should equal r
            rad_constraint = np.linalg.norm(t - c) - r
            # Regularization: prefer small radius (tight tripod)
            small_radius_constraint = r ** 2
            res.append(rad_constraint)
            res.append(small_radius_constraint)
        return res

    initial_guess = np.array([
        initial_center[0],
        initial_center[1],
        initial_center[2],
        0.12  # Initial radius guess (12cm — typical tripod head)
    ])

    bounds = (
        [-np.inf, -np.inf, -np.inf, 0.0],
        [np.inf, np.inf, np.inf, 0.5]  # Max 50cm radius
    )

    result = least_squares(
        residuals, initial_guess,
        bounds=bounds,
        args=(points, directions),
        loss='soft_l1'
    )

    return result.x[:3], result.x[3]


def main():
    parser = argparse.ArgumentParser(
        description="Compute tripod parameters from Pass 1 camera output"
    )
    parser.add_argument('-i', '--input', required=True,
                        help='Path to Pass 1 JSON (e.g. bt_output_8_pts.json)')
    parser.add_argument('-o', '--output', default='tripod.json',
                        help='Output tripod JSON path')
    parser.add_argument('--min-score', type=float, default=0.6,
                        help='Minimum frame score to include')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("TRIPOD PARAMETER ESTIMATION")
    print(f"{'='*60}")

    # 1. Extract camera rays from Pass 1 output
    positions, axes = extract_camera_rays(args.input, args.min_score)
    print(f"  Extracted {len(positions)} well-tracked frames (score > {args.min_score})")

    if len(positions) < 3:
        print("  ERROR: Need at least 3 well-tracked frames for tripod estimation.")
        return

    positions_arr = np.array(positions)
    print(f"  Position range:")
    print(f"    X: [{positions_arr[:, 0].min():.2f}, {positions_arr[:, 0].max():.2f}]")
    print(f"    Y: [{positions_arr[:, 1].min():.2f}, {positions_arr[:, 1].max():.2f}]")
    print(f"    Z: [{positions_arr[:, 2].min():.2f}, {positions_arr[:, 2].max():.2f}]")

    # 2. Compute pairwise optical axis intersections
    intersections = compute_intersection_points(positions, axes)
    print(f"  Computed {len(intersections)} ray-pair intersections")

    if len(intersections) < 3:
        # Fallback for narrow-pan cameras (e.g. cricket sideline):
        # All optical axes point in ~same direction, so pairwise intersections fail.
        # Sphere fitting is also ill-conditioned with parallel axes (produces wrong Z).
        # Just use robust position statistics directly.
        print("  INFO: Too few ray intersections (camera pans very little).")
        print("  Using robust position statistics as tripod estimate.")
        if HAS_SKLEARN and len(positions) >= 4:
            try:
                robust_pos = MinCovDet(random_state=42).fit(positions_arr)
                center = robust_pos.location_
            except Exception:
                center = np.median(positions_arr, axis=0)
        else:
            center = np.median(positions_arr, axis=0)
        # Estimate radius from position scatter (how much the camera "wobbles")
        dists = np.linalg.norm(positions_arr - center, axis=1)
        radius = float(np.median(dists))
        used_sphere_fit = False
    else:
        # 3. Robust mean via MinCovDet on ray intersections
        if HAS_SKLEARN and len(intersections) >= 4:
            try:
                robust_est = MinCovDet(random_state=42).fit(intersections)
                initial_center = robust_est.location_
                print(f"  Robust center estimate (MinCovDet): "
                      f"[{initial_center[0]:.2f}, {initial_center[1]:.2f}, {initial_center[2]:.2f}]")
            except Exception:
                initial_center = np.median(intersections, axis=0)
                print(f"  Median center estimate: "
                      f"[{initial_center[0]:.2f}, {initial_center[1]:.2f}, {initial_center[2]:.2f}]")
        else:
            initial_center = np.median(intersections, axis=0)
            print(f"  Median center estimate: "
                  f"[{initial_center[0]:.2f}, {initial_center[1]:.2f}, {initial_center[2]:.2f}]")

        # 4. Fit sphere (only when intersections provide a good initial guess)
        center, radius = fit_tripod_sphere(positions, axes, initial_center)
        used_sphere_fit = True

    print(f"\n  TRIPOD RESULT {'(sphere fit)' if used_sphere_fit else '(robust mean)'}:")
    print(f"    Center: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
    print(f"    Radius: {radius:.4f} m")

    # 5. Save output
    output_dict = {
        "sphere": {
            "center": center,
            "radius": radius
        },
        "method": "sphere_fit" if used_sphere_fit else "robust_mean",
        "stats": {
            "n_frames_used": len(positions),
            "n_intersections": len(intersections),
            "position_mean": positions_arr.mean(axis=0),
            "position_std": positions_arr.std(axis=0)
        }
    }

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_dict, f, indent=4, cls=NumpyEncoder)

    print(f"\n  Saved to: {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
