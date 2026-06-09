import cv2
import numpy as np
import argparse
import sys
import os

def extract_pitch_edges_from_mask(mask, num_points=80, border_margin=50):
    """
    Given a binary mask of the pitch, extracts sample points along its logical boundaries.
    """
    if mask is None or np.sum(mask) == 0:
        return []

    img_h, img_w = mask.shape[:2]
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    largest_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest_contour)
    epsilon = 0.005 * cv2.arcLength(hull, True)
    approx_poly = cv2.approxPolyDP(hull, epsilon, True)
    
    poly_vertices = approx_poly.reshape(-1, 2)
    n_vertices = len(poly_vertices)
    
    if n_vertices < 3:
        return []
        
    sampled_points = []
    points_per_edge = max(1, num_points // n_vertices)
    
    for i in range(n_vertices):
        p1 = poly_vertices[i].astype(float)
        p2 = poly_vertices[(i + 1) % n_vertices].astype(float)
        
        dx = abs(p2[0] - p1[0])
        dy = abs(p2[1] - p1[1])
        
        at_frame_top = (p1[1] < 5 and p2[1] < 5)
        at_frame_bottom = (p1[1] > img_h - 5 and p2[1] > img_h - 5)
        at_frame_left = (p1[0] < 5 and p2[0] < 5)
        at_frame_right = (p1[0] > img_w - 5 and p2[0] > img_w - 5)
        
        if at_frame_top or at_frame_bottom or at_frame_left or at_frame_right:
            continue
        
        for j in range(points_per_edge):
            alpha = j / points_per_edge
            pt = p1 * (1.0 - alpha) + p2 * alpha
            x, y = int(pt[0]), int(pt[1])
            
            if x < border_margin or x > (img_w - border_margin):
                continue
            if y < border_margin or y > (img_h - border_margin):
                continue
            
            sampled_points.append((x, y))
    
    return sampled_points


def main():
    parser = argparse.ArgumentParser(description="Extract ONLY the SAM predicted pitch edges (red lines and yellow dots) overlaid on the original frame.")
    parser.add_argument('--image', required=True, help='Path to the original frame image (e.g. 000588.jpg)')
    parser.add_argument('--mask', required=True, help='Path to the SAM mask image (e.g. mask_000588.png)')
    parser.add_argument('--output', required=True, help='Path to save the output image')
    args = parser.parse_args()

    # Read image
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Error: Could not read image at {args.image}")
        sys.exit(1)
    
    # Read mask
    mask_raw = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if mask_raw is None:
        print(f"Error: Could not read mask at {args.mask}")
        sys.exit(1)

    h, w = frame.shape[:2]
    
    # Resize mask if needed
    if mask_raw.shape[0] != h or mask_raw.shape[1] != w:
        mask_raw = cv2.resize(mask_raw, (w, h))
        
    mask_binary = (mask_raw > 127).astype(np.uint8) * 255
    
    # Extract edges
    edge_points = extract_pitch_edges_from_mask(mask_binary, num_points=80)
    
    # Visualization: draw only the red polygon and yellow dots
    vis = frame.copy()
    if len(edge_points) > 1:
        pts = np.array(edge_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        
    for (px, py) in edge_points:
        cv2.circle(vis, (int(px), int(py)), 4, (0, 255, 255), -1)

    # Save output
    cv2.imwrite(args.output, vis)
    print(f"File successfully saved to {args.output}")

if __name__ == '__main__':
    main()
