import argparse
import os
import sys
import json
import subprocess
import numpy as np

def run_command(cmd_str):
    print(f"\n[RUNNING] {cmd_str}")
    subprocess.run(cmd_str, shell=True, check=True)

def estimate_tripod(pass1_json_path, output_tripod_json_path):
    print(f"\n[ESTIMATING TRIPOD] Reading {pass1_json_path}...")
    with open(pass1_json_path, 'r') as f:
        data = json.load(f)
        
    x_positions = []
    y_positions = []
    z_positions = []
    
    for frame_name, frame_data in data.items():
        if "cp" in frame_data:
            x_positions.append(frame_data["cp"]["positionXMeters"])
            y_positions.append(frame_data["cp"]["positionYMeters"])
            z_positions.append(frame_data["cp"]["positionZMeters"])
            
    if not x_positions:
        print("Error: Pass 1 yielded no camera positions!")
        return False
        
    median_x = np.median(x_positions)
    median_y = np.median(y_positions)
    median_z = np.median(z_positions)
    
    print(f"Calculated Median Tripod Position: X={median_x:.2f}, Y={median_y:.2f}, Z={median_z:.2f}")
    
    # Must match C++ loadTripodInfo format: {"sphere": {"center": [x,y,z], "radius": r}}
    tripod_data = {
        "sphere": {
            "center": [float(median_x), float(median_y), float(median_z)],
            "radius": 0.0
        }
    }
    
    with open(output_tripod_json_path, 'w') as f:
        json.dump(tripod_data, f, indent=4)
        
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Full 2-Pass BroadTrack Pipeline for Cricket")
    parser.add_argument("-f", "--frames", required=True, help="Path to continuous frames directory")
    parser.add_argument("-m", "--model", required=True, help="Path to YOLO keypoint model")
    parser.add_argument("-o", "--output", default="final_tracked.json", help="Final output JSON path")
    args = parser.parse_args()

    # Step 1: Run Perception Export
    keypoints_json = "temp_keypoints.json"
    run_command(f"{sys.executable} export_cricket_inputs.py -f {args.frames} -m {args.model} -o {keypoints_json}")
    
    # Step 2: C++ Pass 1 (No Tripod)
    pass1_json = "temp_pass1.json"
    cwd = os.getcwd().replace('\\', '/')
    docker_cmd_base = f"docker run --rm -v {cwd}:/workdir cricket_cpp cricket_tracker -f /workdir/{args.frames} -k /workdir/{keypoints_json}"
    run_command(f"{docker_cmd_base} -o /workdir/{pass1_json}")
    
    # Step 3: Estimate Tripod Medians
    tripod_json = "estimated_tripod.json"
    success = estimate_tripod(pass1_json, tripod_json)
    if not success:
        exit(1)
        
    # Step 4: C++ Pass 2 (With Tripod constraint)
    run_command(f"{docker_cmd_base} -t /workdir/{tripod_json} -o /workdir/{args.output}")
    
    print(f"\n[DONE] Pipeline successfully wrote final tracking sequence to {args.output}")
