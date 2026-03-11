"""
Fix Hallucinated Keypoints - Generate Partial Visibility Training Data

Problem: Model predicts 4 corners even when only 2 are visible
Solution: Train with cropped/zoomed images where some corners are off-screen
"""

import cv2
import numpy as np
from pathlib import Path
import random
import shutil


class PartialVisibilityGenerator:
    """
    Generate training data with partial keypoint visibility
    to teach model to handle zoomed/cropped views
    """

    def __init__(self, output_dir="partial_visibility_dataset"):
        self.output_dir = Path(output_dir)
        (self.output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)

    def load_yolo_keypoints(self, label_path, img_w, img_h):
        """Load YOLOv8 pose format. Returns None if label is malformed."""
        with open(label_path) as f:
            line = f.read().strip().split()

        # Need at least: class + 4 bbox + 4 keypoints * 3 = 17 values
        if len(line) < 17:
            return None, None

        nums = [float(x) for x in line[1:]]

        # Bbox
        cx, cy, bw, bh = nums[0:4]

        # Keypoints (x, y, visibility) × 4
        kpts = []
        for i in range(4):
            base = 4 + i * 3
            x = nums[base] * img_w
            y = nums[base + 1] * img_h
            v = nums[base + 2]
            kpts.append([x, y, v])

        return np.array(kpts), (cx, cy, bw, bh)

    def save_yolo_keypoints(self, label_path, kpts, bbox, img_w, img_h):
        """Save YOLOv8 pose format with proper visibility flags"""
        cx, cy, bw, bh = bbox

        parts = ["0", f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]

        for kp in kpts:
            if kp[2] > 0:  # visible
                kx = np.clip(kp[0] / img_w, 0, 1)
                ky = np.clip(kp[1] / img_h, 0, 1)
                parts.extend([f"{kx:.6f}", f"{ky:.6f}", "2"])  # visibility = 2 (visible)
            else:  # not visible
                # IMPORTANT: Still include coordinates but mark as invisible
                parts.extend(["0.0", "0.0", "0"])  # visibility = 0 (not visible)

        with open(label_path, "w") as f:
            f.write(" ".join(parts))

    def update_bbox_from_visible_keypoints(self, kpts, img_w, img_h):
        """Compute bbox from ONLY visible keypoints"""
        visible_kpts = [kp for kp in kpts if kp[2] > 0]

        if len(visible_kpts) == 0:
            return None

        xs = [kp[0] for kp in visible_kpts]
        ys = [kp[1] for kp in visible_kpts]

        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)

        # Add padding
        pad = 30
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(img_w, x2 + pad)
        y2 = min(img_h, y2 + pad)

        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h

        return (cx, cy, bw, bh)

    # ─────────────────────────────────────────────────────
    # CROP STRATEGIES
    # ─────────────────────────────────────────────────────

    def crop_zoom_batting_end(self, img, kpts):
        """
        Zoom into batting end (top of image, far from camera)
        Result: batting corners visible, bowling corners off-screen
        """
        h, w = img.shape[:2]

        # Crop top 60% of image (batting end is at the top)
        crop_h = int(h * 0.6)
        cropped = img[:crop_h, :]

        new_kpts = kpts.copy()
        for kp in new_kpts:
            if kp[1] < 0 or kp[1] >= crop_h:
                kp[2] = 0  # mark invisible

        return cropped, new_kpts

    def crop_zoom_bowling_end(self, img, kpts):
        """
        Zoom into bowling end (bottom of image, near camera)
        Result: bowling corners visible, batting corners off-screen
        """
        h, w = img.shape[:2]

        # Crop bottom 60% of image (bowling end is at the bottom)
        crop_y = int(h * 0.4)
        cropped = img[crop_y:, :]

        new_kpts = kpts.copy()
        for kp in new_kpts:
            kp[1] -= crop_y
            if kp[1] < 0 or kp[1] >= (h - crop_y):
                kp[2] = 0  # mark invisible

        return cropped, new_kpts

    def crop_left_side(self, img, kpts):
        """
        Crop to left side
        Result: left corners visible, right corners off-screen
        """
        h, w = img.shape[:2]

        # Crop left 65%
        crop_w = int(w * 0.65)
        cropped = img[:, :crop_w]

        new_kpts = kpts.copy()
        for kp in new_kpts:
            if kp[0] < 0 or kp[0] >= crop_w:
                kp[2] = 0

        return cropped, new_kpts

    def crop_right_side(self, img, kpts):
        """
        Crop to right side
        Result: right corners visible, left corners off-screen
        """
        h, w = img.shape[:2]

        # Crop right 65%
        crop_x = int(w * 0.35)
        cropped = img[:, crop_x:]

        new_kpts = kpts.copy()
        for kp in new_kpts:
            kp[0] -= crop_x
            if kp[0] < 0 or kp[0] >= (w - crop_x):
                kp[2] = 0

        return cropped, new_kpts

    def crop_center_zoom(self, img, kpts, zoom_factor=1.5):
        """
        Zoom into center of pitch
        Result: potentially ALL corners off-screen
        """
        h, w = img.shape[:2]

        # Calculate crop region
        new_w = int(w / zoom_factor)
        new_h = int(h / zoom_factor)

        x1 = (w - new_w) // 2
        y1 = (h - new_h) // 2

        cropped = img[y1:y1+new_h, x1:x1+new_w]

        # Adjust keypoints
        new_kpts = kpts.copy()
        for kp in new_kpts:
            kp[0] -= x1
            kp[1] -= y1

            # Check if still in bounds
            if kp[0] < 0 or kp[0] >= new_w or kp[1] < 0 or kp[1] >= new_h:
                kp[2] = 0

        return cropped, new_kpts

    def crop_random_window(self, img, kpts, window_size=0.7):
        """
        Random crop window
        Result: variable number of corners visible (0-4)
        """
        h, w = img.shape[:2]

        # Random window size
        crop_w = int(w * random.uniform(window_size, 0.95))
        crop_h = int(h * random.uniform(window_size, 0.95))

        # Random position
        x1 = random.randint(0, w - crop_w)
        y1 = random.randint(0, h - crop_h)

        cropped = img[y1:y1+crop_h, x1:x1+crop_w]

        new_kpts = kpts.copy()
        for kp in new_kpts:
            kp[0] -= x1
            kp[1] -= y1

            if kp[0] < 0 or kp[0] >= crop_w or kp[1] < 0 or kp[1] >= crop_h:
                kp[2] = 0

        return cropped, new_kpts

    def add_random_occlusion(self, img, kpts):
        """
        Add synthetic occlusion blocks to simulate players/umpires covering the pitch.
        Result: some corners may be hidden.
        """
        h, w = img.shape[:2]
        occluded = img.copy()
        new_kpts = kpts.copy()

        for _ in range(random.randint(1, 4)):
            x1 = random.randint(0, max(1, w - 100))
            y1 = random.randint(0, max(1, h - 100))
            x2 = x1 + random.randint(100, 250)
            y2 = y1 + random.randint(100, 250)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.rectangle(occluded, (x1, y1), (x2, y2), color, -1)

            # Mark keypoints inside the box as invisible
            for kp in new_kpts:
                if kp[2] > 0 and x1 <= kp[0] <= x2 and y1 <= kp[1] <= y2:
                    kp[2] = 0

        return occluded, new_kpts

    # ─────────────────────────────────────────────────────
    # GENERATE AUGMENTED DATASET
    # ─────────────────────────────────────────────────────

    def generate_partial_visibility_samples(self, img, kpts, n_samples=6):
        """
        Generate multiple cropped versions with different visibility patterns
        """
        samples = []

        # Strategy distribution
        strategies = [
            ("batting_end", lambda: self.crop_zoom_batting_end(img, kpts)),
            ("bowling_end", lambda: self.crop_zoom_bowling_end(img, kpts)),
            ("left_side", lambda: self.crop_left_side(img, kpts)),
            ("right_side", lambda: self.crop_right_side(img, kpts)),
            ("center_zoom", lambda: self.crop_center_zoom(img, kpts, 1.4)),
            ("center_zoom2", lambda: self.crop_center_zoom(img, kpts, 1.8)),
            ("random1", lambda: self.crop_random_window(img, kpts, 0.6)),
            ("random2", lambda: self.crop_random_window(img, kpts, 0.7)),
            ("occlusion", lambda: self.add_random_occlusion(img, kpts)),
            ("crop_occlusion", lambda: self.crop_random_window(*self.add_random_occlusion(img, kpts), 0.7)),
        ]

        # Generate samples
        selected = random.sample(strategies, min(n_samples, len(strategies)))

        for name, strategy_fn in selected:
            try:
                crop_img, crop_kpts = strategy_fn()

                # Count visible keypoints
                n_visible = sum(1 for kp in crop_kpts if kp[2] > 0)

                # Only save if 0-3 keypoints visible (not 4)
                if 0 <= n_visible <= 3:
                    samples.append((name, crop_img, crop_kpts))

            except Exception as e:
                print(f"  ⚠ Strategy {name} failed: {e}")

        return samples

    def process_dataset(self, images_dir, labels_dir, n_samples_per_image=2):
        """
        Process full dataset to generate partial visibility training data
        """
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)

        img_files = list(images_dir.glob("*.jpg")) + \
                    list(images_dir.glob("*.png"))

        print(f"\n{'='*60}")
        print("GENERATING PARTIAL VISIBILITY DATASET")
        print(f"{'='*60}")
        print(f"Input: {len(img_files)} images")
        print(f"Samples per image: {n_samples_per_image}")

        total_saved = 0
        visibility_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        for img_path in img_files:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]

            # Load label
            lbl_path = labels_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue

            kpts, bbox = self.load_yolo_keypoints(lbl_path, w, h)
            
            # Skip malformed labels or images where not all 4 are visible
            if kpts is None:
                print(f"  Skipping {img_path.name}: malformed label")
                continue
            
            # Only generate crops from images where all 4 keypoints are originally visible
            n_visible_orig = sum(1 for kp in kpts if kp[2] > 0)
            if n_visible_orig < 4:
                # Still copy the original (it's valid partial-visibility training data already!)
                shutil.copy2(img_path,
                    self.output_dir / "images" / "train" / img_path.name)
                shutil.copy2(lbl_path,
                    self.output_dir / "labels" / "train" / lbl_path.name)
                total_saved += 1
                continue

            # Keep original (all 4 visible)
            shutil.copy2(img_path,
                self.output_dir / "images" / "train" / img_path.name)
            shutil.copy2(lbl_path,
                self.output_dir / "labels" / "train" / lbl_path.name)
            total_saved += 1

            # Generate partial visibility samples
            samples = self.generate_partial_visibility_samples(
                img, kpts, n_samples_per_image
            )

            for i, (strategy_name, crop_img, crop_kpts) in enumerate(samples):
                crop_h, crop_w = crop_img.shape[:2]

                # Count visible
                n_visible = sum(1 for kp in crop_kpts if kp[2] > 0)
                visibility_counts[n_visible] = visibility_counts.get(n_visible, 0) + 1

                # Save image
                crop_name = f"{img_path.stem}_{strategy_name}{img_path.suffix}"
                crop_img_path = self.output_dir / "images" / "train" / crop_name
                cv2.imwrite(str(crop_img_path), crop_img)

                # Save label with updated visibility
                new_bbox = self.update_bbox_from_visible_keypoints(
                    crop_kpts, crop_w, crop_h
                )

                if new_bbox is not None:
                    crop_lbl_path = self.output_dir / "labels" / "train" / \
                                   f"{img_path.stem}_{strategy_name}.txt"
                    self.save_yolo_keypoints(
                        crop_lbl_path, crop_kpts, new_bbox, crop_w, crop_h
                    )
                    total_saved += 1

        print(f"\n✓ Generated {total_saved} total images")
        print(f"\nVisibility distribution:")
        print(f"  1 corner visible:  {visibility_counts.get(1, 0)} images")
        print(f"  2 corners visible: {visibility_counts.get(2, 0)} images")
        print(f"  3 corners visible: {visibility_counts.get(3, 0)} images")
        print(f"  4 corners visible: {len(img_files)} images (originals)")
        print(f"\nSaved to: {self.output_dir}")


# ─────────────────────────────────────────────────────────────
# POST-PROCESSING: Filter Low-Confidence Invisible Keypoints
# ─────────────────────────────────────────────────────────────

def filter_invisible_keypoints(results, confidence_threshold=0.3):
    """
    Post-process YOLO results to mark low-confidence keypoints as invisible

    Args:
        results: YOLOv8 results object
        confidence_threshold: Min confidence for visible keypoint

    Returns:
        Filtered keypoints with proper visibility flags
    """
    if results[0].keypoints is None:
        return None

    kpts = results[0].keypoints.xy[0].cpu().numpy()  # (4, 2)
    confs = results[0].keypoints.conf[0].cpu().numpy()  # (4,)

    filtered_kpts = []

    for i, (kpt, conf) in enumerate(zip(kpts, confs)):
        if conf > confidence_threshold:
            filtered_kpts.append({
                'xy': kpt,
                'confidence': conf,
                'visible': True,
                'index': i
            })
        else:
            filtered_kpts.append({
                'xy': kpt,
                'confidence': conf,
                'visible': False,
                'index': i
            })

    return filtered_kpts


# ─────────────────────────────────────────────────────────────
# VISUALIZATION WITH VISIBILITY
# ─────────────────────────────────────────────────────────────

def visualize_with_visibility(image, keypoints, show_invisible=True):
    """
    Draw keypoints with different styles for visible/invisible
    """
    vis = image.copy()

    labels = ["BatL", "BatR", "BowR", "BowL"]
    colors = [(0, 255, 0), (0, 255, 0), (255, 165, 0), (255, 165, 0)]

    for kpt_info, label, color in zip(keypoints, labels, colors):
        pt = tuple(kpt_info['xy'].astype(int))
        conf = kpt_info['confidence']
        visible = kpt_info['visible']

        if visible:
            # Solid circle for visible
            cv2.circle(vis, pt, 8, color, -1)
            cv2.circle(vis, pt, 10, color, 2)
            text = f"{label} {conf:.2f}"
            cv2.putText(vis, text, (pt[0]+12, pt[1]-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        elif show_invisible:
            # Dashed circle for invisible (low confidence)
            cv2.circle(vis, pt, 8, (100, 100, 100), 1)
            text = f"{label} {conf:.2f}?"
            cv2.putText(vis, text, (pt[0]+12, pt[1]-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # Add visibility count
    n_visible = sum(1 for kpt in keypoints if kpt['visible'])
    cv2.putText(vis, f"Visible corners: {n_visible}/4",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return vis


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("\nUsage:")
        print("  python fix_hallucination.py <images_dir> <labels_dir>")
        print("\nExample:")
        print("  python fix_hallucination.py \\")
        print("    cricket_dataset/images/train \\")
        print("    cricket_dataset/labels/train")
        print("\nThis will generate:")
        print("  - Original images (4 corners visible)")
        print("  - Zoomed batting end (2 corners visible)")
        print("  - Zoomed bowling end (2 corners visible)")
        print("  - Left side crops (2 corners visible)")
        print("  - Right side crops (2 corners visible)")
        print("  - Center zoom (0-2 corners visible)")
        print("  - Random crops (0-3 corners visible)")
        print("  - Synthetic occlusion (0-3 corners visible)")
        print("\n~7x more training data with partial visibility and full occlusions!")
    else:
        gen = PartialVisibilityGenerator("BroadTrack/cricket/partial_visibility_dataset")
        gen.process_dataset(
            images_dir=sys.argv[1],
            labels_dir=sys.argv[2],
            n_samples_per_image=6
        )

        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("\n1. Create data.yaml for new dataset")
        print("\n2. Retrain with partial visibility data:")
        print("   python train_cricket_yolo.py --train \\")
        print("     --data partial_visibility_dataset/data.yaml \\")
        print("     --epochs 150 \\")
        print("     --model yolov8s-pose.pt")
        print("\n3. Test on zoomed images - model should now output")
        print("   only visible corners!")