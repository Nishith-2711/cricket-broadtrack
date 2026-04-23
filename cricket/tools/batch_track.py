"""
Batch Tracking Pipeline
=======================
Processes all videos in input_videos/ through the full 3-step pipeline:
  Step 1: Pass 1 (Initial Pose Discovery)
  Step 2: Compute Tripod Pivot
  Step 3: Pass 2 (Locked Tracking)

Usage:
  python batch_track.py
  python batch_track.py --input-dir input_videos --output-dir output_jsons
  python batch_track.py --videos input1.mp4 input3.mp4   # specific videos only
"""

import os
import sys
import glob
import time
import argparse
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable  # Use the same Python interpreter


def run_step(description, cmd, cwd=SCRIPT_DIR):
    """Run a subprocess and print real-time output. Returns True on success."""
    print(f"\n{'─'*60}")
    print(f"  {description}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'─'*60}")

    start = time.time()
    result = subprocess.run(cmd, cwd=cwd)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  ❌ FAILED (exit code {result.returncode}) [{elapsed:.1f}s]")
        return False
    else:
        print(f"  ✅ Done [{elapsed:.1f}s]")
        return True


def process_video(video_path, output_dir, model_path, seg_model_path=None, skip_existing=False):
    """Run the full 3-step pipeline for a single video."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Output file paths
    pass1_json = os.path.join(output_dir, f"{video_name}_pass1.json")
    tripod_json = os.path.join(output_dir, f"{video_name}_tripod.json")
    final_json = os.path.join(output_dir, f"{video_name}_final.json")

    print(f"\n{'='*60}")
    print(f"  PROCESSING: {video_name}")
    print(f"{'='*60}")

    # ── Step 1: Pass 1 ──
    if skip_existing and os.path.exists(pass1_json):
        print(f"  ⏭️  Skipping Pass 1 (already exists: {pass1_json})")
    else:
        cmd = [
            PYTHON, "broadtrack_cricket.py",
            "-v", video_path,
            "-m", model_path,
            "-o", pass1_json,
        ]
        if not run_step(f"Step 1/3: Pass 1 — {video_name}", cmd):
            return False, video_name

    # ── Step 2: Compute Tripod ──
    if skip_existing and os.path.exists(tripod_json):
        print(f"  ⏭️  Skipping Tripod (already exists: {tripod_json})")
    else:
        cmd = [
            PYTHON, "compute_tripod.py",
            "-i", pass1_json,
            "-o", tripod_json,
        ]
        if not run_step(f"Step 2/3: Compute Tripod — {video_name}", cmd):
            return False, video_name

    # ── Step 3: Pass 2 (Locked Tracking) ──
    if skip_existing and os.path.exists(final_json):
        print(f"  ⏭️  Skipping Pass 2 (already exists: {final_json})")
    else:
        cmd = [
            PYTHON, "broadtrack_cricket.py",
            "-v", video_path,
            "-m", model_path,
            "--tripod", tripod_json,
            "-o", final_json,
        ]
        # Add seg model if provided
        if seg_model_path:
            cmd.extend(["-s", seg_model_path, "--use-lines"])

        if not run_step(f"Step 3/3: Pass 2 (Locked) — {video_name}", cmd):
            return False, video_name

    return True, video_name


def main():
    parser = argparse.ArgumentParser(description="Batch tracking pipeline for all videos")
    parser.add_argument("--input-dir", default="input_videos",
                        help="Directory containing input videos (default: input_videos)")
    parser.add_argument("--output-dir", default="output_jsons",
                        help="Directory for output JSONs (default: output_jsons)")
    parser.add_argument("--model", default=os.path.join("..", "models", "keypoint_v5.pt"),
                        help="Path to YOLO pose model")
    parser.add_argument("--seg-model", default=None,
                        help="Path to segmentation model (optional, omit to skip seg)")
    parser.add_argument("--videos", nargs="+", default=None,
                        help="Specific video filenames to process (e.g. input1.mp4 input3.mp4)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip steps whose output files already exist")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Discover videos
    if args.videos:
        video_files = [os.path.join(args.input_dir, v) for v in args.videos]
    else:
        video_files = sorted(
            glob.glob(os.path.join(args.input_dir, "*.mp4")) +
            glob.glob(os.path.join(args.input_dir, "*.avi")) +
            glob.glob(os.path.join(args.input_dir, "*.mkv"))
        )

    if not video_files:
        print(f"No videos found in {args.input_dir}/")
        return

    print(f"\n{'='*60}")
    print(f"  BATCH TRACKING PIPELINE")
    print(f"{'='*60}")
    print(f"  Videos found:    {len(video_files)}")
    print(f"  Model:           {args.model}")
    print(f"  Seg model:       {args.seg_model or '(none — keypoints only)'}")
    print(f"  Output dir:      {args.output_dir}")
    print(f"  Skip existing:   {args.skip_existing}")
    print()
    for v in video_files:
        print(f"    • {os.path.basename(v)}")

    total_start = time.time()
    results = []

    for i, video_path in enumerate(video_files, 1):
        print(f"\n{'#'*60}")
        print(f"  VIDEO {i}/{len(video_files)}")
        print(f"{'#'*60}")
        success, name = process_video(
            video_path, args.output_dir, args.model,
            seg_model_path=args.seg_model,
            skip_existing=args.skip_existing
        )
        results.append((name, success))

    total_elapsed = time.time() - total_start

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  BATCH COMPLETE — {total_elapsed:.1f}s total")
    print(f"{'='*60}")

    passed = sum(1 for _, s in results if s)
    failed = sum(1 for _, s in results if not s)

    for name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status}  {name}")

    print(f"\n  Passed: {passed}/{len(results)}    Failed: {failed}/{len(results)}")
    print()


if __name__ == "__main__":
    main()
