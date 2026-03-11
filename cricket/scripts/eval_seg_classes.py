from ultralytics import YOLO
import sys
import os

paths = [
    r"C:\Codes\Projects\Football_analysis\models\best.pt",
    r"C:\Codes\Projects\Football_analysis\models\best (1).pt"
]

with open("eval_out_utf8.txt", "w", encoding="utf-8") as f:
    for p in paths:
        if not os.path.exists(p):
            f.write(f"\nModel: {p} not found\n")
            continue
            
        try:
            m = YOLO(p)
            f.write(f"\nModel: {p}\n")
            f.write(f"Classes: {m.names}\n")
            f.write(f"Task: {m.task}\n")
        except Exception as e:
            f.write(f"Failed to load {p}: {e}\n")
