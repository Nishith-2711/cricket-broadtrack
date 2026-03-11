
import cv2
import argparse
from ultralytics import YOLO
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', required=True, help='Path to image')
    parser.add_argument('-m', '--model', required=True, help='Path to YOLO pose model')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Load model
    model = YOLO(args.model)
    
    # Run inference
    results = model(args.image, conf=args.conf)
    r = results[0]
    
    img = cv2.imread(args.image)
    
    if r.keypoints is not None and len(r.keypoints.xy) > 0:
        kpts = r.keypoints.xy[0].cpu().numpy()
        confs = r.keypoints.conf[0].cpu().numpy() if r.keypoints.conf is not None else [1.0]*len(kpts)
        
        print(f"Detected {len(kpts)} keypoints:")
        
        for i, (pt, conf) in enumerate(zip(kpts, confs)):
            x, y = int(pt[0]), int(pt[1])
            print(f"  Point {i}: ({x}, {y}) Conf: {conf:.2f}")
            
            # Draw point
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            
            # Draw label with index
            label = f"{i}"
            cv2.putText(img, label, (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, (0, 255, 255), 2)  # Yellow text
    else:
        print("No keypoints detected.")

    # Save output
    outfile = "yolo_debug.jpg"
    cv2.imwrite(outfile, img)
    print(f"Saved debug image to {outfile}")

if __name__ == "__main__":
    main()
