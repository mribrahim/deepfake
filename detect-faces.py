import os
import argparse
import torch
import cv2
from pathlib import Path

from FaceBoxesV2.faceboxes_detector import FaceBoxesDetector


def detections_to_boxes(detections, det_box_scale, image_width, image_height):
    boxes = []
    if detections is not None:
        for det in detections:
            det_xmin = det[2]
            det_ymin = det[3]
            det_width = det[4]
            det_height = det[5]
            det_xmax = det_xmin + det_width - 1
            det_ymax = det_ymin + det_height - 1

            det_xmin -= int(det_width * (det_box_scale - 1) / 2)
            det_ymin += int(det_height * (det_box_scale - 1) / 2)
            det_xmax += int(det_width * (det_box_scale - 1) / 2)
            det_ymax += int(det_height * (det_box_scale - 1) / 2)
            det_xmin = max(det_xmin, 0)
            det_ymin = max(det_ymin, 0)
            det_xmax = min(det_xmax, image_width - 1)
            det_ymax = min(det_ymax, image_height - 1)
            
            # Recompute the width and height if needed
            det_width = det_xmax - det_xmin + 1
            det_height = det_ymax - det_ymin + 1
            
            boxes.append([det_xmin, det_ymin, det_xmax, det_ymax])
            
    return boxes


def process_image(image_path, detector, my_thresh, det_box_scale):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[-] Warning: Could not read image {image_path}")
        return

    image_height, image_width = image.shape[:2]

    detections, _ = detector.detect(image, my_thresh, 1)
    boxes = detections_to_boxes(detections, det_box_scale, image_width, image_height)

    if len(boxes) == 0:
        print(f"[-] Warning: No face detected in {image_path}. Removing corrupted/unusable file.")
        try:
            os.remove(image_path)  # Remove the file if no face is visible
        except Exception as e:
            print(f"[-] Error removing file {image_path}: {e}")
        return

    # Use the first detected face (highest confidence ROI)
    box = boxes[0]
    x1, y1, x2, y2 = map(int, box)
    face_patch = image[y1:y2, x1:x2]

    # Overwrite the original source file with the extracted face patch matrix context
    cv2.imwrite(image_path, face_patch)


def main():
    parser = argparse.ArgumentParser(description="In-place Face ROI Crop Preprocessing Script")
    parser.add_argument("--root_dir", type=str, default="sample_dataset/", 
                        help="Path to root target folder containing images or subdirectories")
    parser.add_argument("--thresh", type=float, default=0.8, help="FaceBoxes detection threshold score")
    parser.add_argument("--box_scale", type=float, default=1.0, help="Bounding box scale factor modifier")
    args = parser.parse_args()

    # Hardware resource configuration validation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing FaceBoxes inference on device: {device}")
    
    # Instantiate framework backbone parameters
    detector = FaceBoxesDetector('FaceBoxes', 'FaceBoxesV2/weights/FaceBoxesV2.pth', True, device)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    root_path = Path(args.root_dir)

    if not root_path.exists():
        print(f"[-] Error: Target root directory '{args.root_dir}' does not exist.")
        return

    print(f"[*] Beginning recursive pre-processing walkthrough inside: {root_path}")
    processed_count = 0
    error_count = 0
        
    for current_path in root_path.rglob('*'):
        if current_path.is_file():
            if current_path.suffix.lower() in image_extensions:
                try:
                    image_path = str(current_path)
                    process_image(image_path, detector, args.thresh, args.box_scale)
                    processed_count += 1
                    print(f"✓ Processed and overwritten Face ROI: {image_path}")
                except Exception as e:
                    error_count += 1
                    print(f"✗ Error processing tracking array context {current_path}: {e}")

    print(f"\n[+] Processing run finalized. Metrics summary: {processed_count} files altered, {error_count} operational faults.")


if __name__ == "__main__":
    main()