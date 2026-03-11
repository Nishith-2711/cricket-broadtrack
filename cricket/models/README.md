# Model Weights

The trained model weights are not included in this repository because they are large files.

Please download or place the required models inside this directory before running the pipeline.

Expected structure:

models/
├── keypoint_v5.pt
├── pitch_seg_best.pt

## Description

* **keypoint_v5.pt** – YOLO model for detecting cricket pitch keypoints
* **pitch_seg_best.pt** – pitch segmentation model

Once the models are placed in this folder, the main pipeline can be executed normally.
