import os
import sys
import glob
import subprocess
import argparse

def run_command(cmd, desc):
    print(f"\n============================================================")
    print(f"[{desc}] Running Command:")
    print(f"  {' '.join(cmd)}")
    print(f"============================================================\n")
    
    # Run the command
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {result.returncode}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Master Controller for BroadTrack SAM3 2-Pass Pipeline")
    parser.add_argument("--videos-dir", type=str, default="input_videos", help="Directory containing input videos")
    parser.add_argument("--masks-dir", type=str, default="final_sam_outputs/output_masks", help="Directory containing SAM mask subfolders")
    parser.add_argument("--models", type=str, default="../models/keypoint_v5.pt", help="Path to YOLO keypoint model")
    parser.add_argument("--out-json", type=str, default="sam_output_json", help="Output directory for JSON result files")
    parser.add_argument("--out-vis", type=str, default="sam_vis", help="Output directory for visualization frames")
    parser.add_argument("--python-exec", type=str, default=sys.executable, help="Python executable to use (defaults to current env)")
    
    args = parser.parse_args()

    # Ensure output directories exist
    os.makedirs(args.out_json, exist_ok=True)
    os.makedirs(args.out_vis, exist_ok=True)

    # Find all videos in the videos directory
    search_pattern = os.path.join(args.videos_dir, "*.mp4")
    video_files = glob.glob(search_pattern)

    if not video_files:
        print(f"No .mp4 videos found in {args.videos_dir}")
        return

    print(f"Found {len(video_files)} videos to process.")

    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n────────────────────────────────────────────────────────────")
        print(f"PROCESSING VIDEO: {video_name}")
        print(f"────────────────────────────────────────────────────────────")

        # Assume mask folder naming convention: video_name + "_masks"
        mask_folder = os.path.join(args.masks_dir, f"{video_name}_masks")
        if not os.path.isdir(mask_folder):
            print(f"[SKIP] Mask folder not found: {mask_folder}. Did you run SAM extract for this video?")
            continue

        # File paths
        pass1_json = os.path.join(args.out_json, f"{video_name}_pass1.json")
        tripod_json = os.path.join(args.out_json, f"{video_name}_tripod.json")
        pass2_json = os.path.join(args.out_json, f"{video_name}_pass2.json")
        vis_out_dir = os.path.join(args.out_vis, f"{video_name}")

        # ---------------------------------------------------------------------
        # PASS 1: Initialize and Track freely (Z-locked after frame 1)
        # ---------------------------------------------------------------------
        pass1_cmd = [
            args.python_exec, "sam_tracker/broadtrack_sam.py",
            "-v", video_path,
            "-m", args.models,
            "--sam-masks", mask_folder,
            "-o", pass1_json
        ]
        
        if not run_command(pass1_cmd, f"{video_name} - PASS 1"):
            print(f"Aborting pipeline for {video_name} due to Pass 1 failure.")
            continue

        # ---------------------------------------------------------------------
        # TRIPOD ESTIMATION: Compute the robust tripod center
        # ---------------------------------------------------------------------
        tripod_cmd = [
            args.python_exec, "compute_tripod.py",
            "-i", pass1_json,
            "-o", tripod_json
        ]

        if not run_command(tripod_cmd, f"{video_name} - TRIPOD"):
            print(f"Aborting pipeline for {video_name} due to Tripod solve failure.")
            continue

        # ---------------------------------------------------------------------
        # PASS 2: Rigid Tracking with Tripod Lock + Visualization
        # ---------------------------------------------------------------------
        pass2_cmd = [
            args.python_exec, "sam_tracker/broadtrack_sam.py",
            "-v", video_path,
            "-m", args.models,
            "--sam-masks", mask_folder,
            "-o", pass2_json,
            "--tripod", tripod_json,
            "--visualize",
            "--vis-dir", vis_out_dir
        ]

        if not run_command(pass2_cmd, f"{video_name} - PASS 2"):
            print(f"Aborting pipeline for {video_name} due to Pass 2 failure.")
            continue

        print(f"\n SUCCESSFULLY COMPLETED: {video_name}")
        print(f"   - Tracking Data: {pass2_json}")
        print(f"   - Visualizations: {vis_out_dir}")

if __name__ == "__main__":
    main()
