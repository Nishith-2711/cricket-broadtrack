"""
Diagnostic video: visualize projected pitch vs segmentation mask
Outputs a full video instead of a single frame
"""

import cv2
import json
import numpy as np
import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True, help="Path to input video")
parser.add_argument("--json", default="bt_output_8_pts.json")
parser.add_argument("--output", default="pitch_iou_diagnostic.mp4")
args = parser.parse_args()

video_path = args.video
json_path = args.json
output_path = args.output

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cricket.src.camera import project_points_batch, focal_from_hfov
from cricket.src.camera_tracker import PITCH_CORNERS_3D

# Load JSON
with open(json_path, "r") as f:
    data = json.load(f)

# Load segmentation model
from ultralytics import YOLO
seg_model = YOLO("..\\models\\pitch_seg_best.pt")

cap = cv2.VideoCapture(video_path)

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Frames:", total_frames)

# Video writer
out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (frame_w, frame_h)
)

pp = np.array([frame_w/2, frame_h/2])

frame_idx = 0

while True:

    ret, frame = cap.read()
    if not ret:
        break

    key = f"/workspace/cricket/frame_{frame_idx:06d}.jpg"

    if key not in data:
        out.write(frame)
        frame_idx += 1
        continue

    entry = data[key]
    cp = entry["cp"]

    rvec = np.array(cp["rvec"])
    position = np.array([
        cp["positionXMeters"],
        cp["positionYMeters"],
        cp["positionZMeters"]
    ])

    hfov = cp["horizontalFieldOfViewDegrees"]
    focal = focal_from_hfov(hfov, frame_w)

    params = np.zeros(8)
    params[0:3] = rvec
    params[3:6] = position
    params[6] = focal
    params[7] = 0

    # Project pitch points
    projected, valid = project_points_batch(params, PITCH_CORNERS_3D, pp)

    corners_4 = projected[:4].astype(np.int32)

    proj_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.fillPoly(proj_mask, [corners_4], 255)

    # Segmentation
    seg_results = seg_model(frame, conf=0.25, verbose=False)

    seg_mask_binary = np.zeros((frame_h, frame_w), dtype=np.uint8)

    sr = seg_results[0]

    if sr.masks is not None and len(sr.masks.data) > 0:

        mask_raw = sr.masks.data[0].cpu().numpy()
        mask_resized = cv2.resize(mask_raw, (frame_w, frame_h))
        seg_mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

    # IoU
    intersection = np.sum((proj_mask > 0) & (seg_mask_binary > 0))
    union = np.sum((proj_mask > 0) | (seg_mask_binary > 0))
    iou = intersection / union if union > 0 else 0

    vis = frame.copy()

    red = np.zeros_like(vis)
    red[:, :, 2] = seg_mask_binary

    green = np.zeros_like(vis)
    green[:, :, 1] = proj_mask

    vis = cv2.addWeighted(vis, 1, red, 0.4, 0)
    vis = cv2.addWeighted(vis, 1, green, 0.4, 0)

    cv2.polylines(vis, [corners_4], True, (0,255,0), 2)

    for i, pt in enumerate(projected):
        cv2.circle(vis, (int(pt[0]), int(pt[1])), 5, (255,255,0), -1)

    cv2.putText(
        vis,
        f"IoU: {iou:.3f}",
        (30,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,255),
        2
    )

    out.write(vis)

    if frame_idx % 50 == 0:
        print("Frame", frame_idx)

    frame_idx += 1


cap.release()
out.release()

print("\nSaved video:", output_path)