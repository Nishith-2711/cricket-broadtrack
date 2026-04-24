# Cricket BroadTrack

Camera tracking system for cricket broadcast video. Detects and tracks the cricket pitch across frames using YOLO keypoint detection and a soft-tripod camera optimizer, with SAM3 segmentation for edge-based tracking when keypoints are partially occluded.

---

## How It Works

The pipeline runs in two passes:

**Pass 1 — Free Tracking**
- YOLO detects up to 8 pitch keypoints per frame (bowling/batting crease corners + popping crease corners)
- PnP solves the initial camera pose from the first frameL
- A per-frame optimizer (scipy `least_squares`, trust-region) minimizes reprojection error across all 8 camera parameters: rotation (angle-axis), position, focal length, and radial distortion
- Camera position is soft-constrained to the previous frame to simulate a tripod

**Tripod Estimation**
- Median camera position across Pass 1 is used to estimate the tripod pivot point

**Pass 2 — Constrained Tracking**
- Same optimizer but with the camera position hard-locked to the estimated tripod sphere center
- Produces stable, drift-free tracking

**SAM3 Edge Tracking**
- Pre-computed SAM3 binary masks of the pitch provide edge-based residuals alongside keypoints
- `LINES_PRIMARY` mode: 1–2 keypoints + edges; `LINES_ONLY` mode: 0 keypoints, edges only
- Line weight is dynamically scaled: 1.0 when ≥3 keypoints are visible, 3.0 when fewer

---

## Pitch Coordinate System

Origin at pitch center. All units in metres.

```
          Batting End (-X)          Bowling End (+X)
               |                         |
  Left (-Y) ───┼─────────────────────────┼─── Right (+Y)
               |                         |

Keypoints (YOLO output order):
  0: batting_left_corner   [-10.06, -1.525, 0]
  1: batting_right_corner  [-10.06,  1.525, 0]
  2: bowling_right_corner  [ 10.06,  1.525, 0]
  3: bowling_left_corner   [ 10.06, -1.525, 0]
  4: batting_left_pop      [ -8.84, -1.525, 0]
  5: batting_right_pop     [ -8.84,  1.525, 0]
  6: bowling_right_pop     [  8.84,  1.525, 0]
  7: bowling_left_pop      [  8.84, -1.525, 0]
```

---

## Project Structure

```
cricket/
├── src/broadtrack/                      # Core library
│   ├── config.py                        # Pitch geometry constants
│   ├── camera.py                        # Camera projection model (angle-axis + distortion)
│   ├── camera_tracker_sam.py            # SAM-augmented optimizer (lines-primary mode)
│   ├── broadtrack_sam.py                # SAM-augmented tracking pipeline
│   ├── compute_tripod.py                # Tripod pivot estimation between passes
│   ├── extract_sam_prediction_frame.py  # Edge extraction from SAM masks
│   ├── sam_detector.py                  # SAM3 offline mask generation
│   ├── master_controller.py             # Batch subprocess orchestrator (SAM pipeline)
│   └── __init__.py
│
├── scripts/
│   ├── evaluate_pipeline.py             # Physical sanity checks + reprojection error + line IoU
│   ├── generate_init_homography.py      # Homography-based initialization (experimental)
│   └── generate_init_optical_flow.py    # Optical flow initialization (experimental)
│
├── tools/
│   ├── visualize_output.py              # Render tracked wireframe onto video frames
│   ├── visualize_pitch.py               # Pitch projection visualization
│   └── visualize_points.py              # Keypoint visualization
│
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

---

## Setup

**Requirements:** Python 3.10+, CUDA-capable GPU (for SAM3)

```bash
pip install -r requirements.txt
pip install -e .
```

Key dependencies:
- `ultralytics` — YOLO keypoint detection
- `opencv-python` — video I/O and image processing
- `scipy` — camera parameter optimization
- `scikit-learn` — robust tripod estimation (MinCovDet)
- `torch` — required by YOLO and SAM3
- `sam3` — SAM3 segmentation (installed from source, see `requirements.txt`)

**Models required** (not included in this repo):
- `models/keypoint_v5.pt` — YOLO keypoint model trained on cricket pitch keypoints
- SAM3 model weights — downloaded automatically on first use

---

## Usage

All commands should be run from the `cricket/` directory.

### 2-Pass Pipeline — Manual (SAM-augmented)

This is the primary way to run the tracker. Run each step individually for a single video.

**Step 1 — Generate SAM masks (offline, once per video)**
```bash
python src/broadtrack/sam_detector.py \
  -v input_videos/input1.mp4 \
  --out-dir final_sam_outputs/output_masks/input1_masks
```

**Step 2 — Pass 1: Free tracking (no tripod constraint)**
```bash
python src/broadtrack/broadtrack_sam.py \
  -v input_videos/input1.mp4 \
  -m models/keypoint_v5.pt \
  --sam-masks final_sam_outputs/output_masks/input1_masks \
  -o output_jsons/input1_pass1.json
```

**Step 3 — Estimate tripod from Pass 1**
```bash
python src/broadtrack/compute_tripod.py \
  -i output_jsons/input1_pass1.json \
  -o output_jsons/input1_tripod.json
```

**Step 4 — Pass 2: Locked tracking with tripod constraint**
```bash
python src/broadtrack/broadtrack_sam.py \
  -v input_videos/input1.mp4 \
  -m models/keypoint_v5.pt \
  --sam-masks final_sam_outputs/output_masks/input1_masks \
  --tripod output_jsons/input1_tripod.json \
  -o output_jsons/input1_final.json \
  --visualize --vis-dir sam_vis/input1
```

---

### Batch — All videos automatically (SAM pipeline)

Runs Pass 1 → tripod → Pass 2 for every video in a directory.

> **Pre-requisite:** SAM masks must be generated for each video before running this. The controller expects a mask subfolder named `<video_name>_masks` inside `--masks-dir` and will skip any video whose mask folder is missing. Run `sam_detector.py` for each video first (see Step 1 above).

```bash
python src/broadtrack/master_controller.py \
  --videos-dir input_videos \
  --masks-dir final_sam_outputs/output_masks \
  --models models/keypoint_v5.pt \
  --out-json output_jsons \
  --out-vis sam_vis
```

### Evaluate Tracking Quality

Runs physical sanity checks and reprojection error aggregation. Pass `--seg-model` or `--sam-masks` to also compute Line IoU:

```bash
# Sanity + reprojection error only
python scripts/evaluate_pipeline.py \
  -j output_jsons/input1_final.json

# With Line IoU (requires seg model)
python scripts/evaluate_pipeline.py \
  -j output_jsons/input1_final.json \
  -v input_videos/input1.mp4 \
  -s models/pitch_seg_best.pt
```

---

## Output Format

Each pipeline produces a JSON file with one entry per frame:

```json
{
  "frame_0001": {
    "mode": "OPTIMIZED",
    "cp": {
      "positionXMeters": -25.3,
      "positionYMeters": 4.1,
      "positionZMeters": 8.7,
      "rvec": [0.12, -0.34, 0.01],
      "focal": 1420.5,
      "k1": -0.02
    }
  }
}
```

`mode` indicates the tracking state for that frame:

| Mode | Meaning |
|---|---|
| `OPTIMIZED` | Full optimizer converged on keypoints |
| `LINES_PRIMARY` | 1–2 keypoints + SAM edges |
| `LINES_ONLY` | 0 keypoints, SAM edges only |
| `COAST` | No detection — last valid pose held |
| `INIT_FAILED` | Could not initialise (insufficient keypoints) |

---

## Camera Model

8-parameter model per frame: `[ax, ay, az, px, py, pz, focal, k1]`

| Parameter | Description |
|---|---|
| `ax, ay, az` | Angle-axis rotation vector |
| `px, py, pz` | Camera position in world coordinates (metres) |
| `focal` | Focal length in pixels |
| `k1` | Radial distortion coefficient |

Projection pipeline: world point → translate → rotate → normalize → distort → pixel

---

## Acknowledgements

- [BroadTrack](https://github.com/evs-broadcast/BroadTrack) 