"""
═══════════════════════════════════════════════════════════════════════════════
INFERENCE
═══════════════════════════════════════════════════════════════════════════════
Runs wound detection on a single image or folder of images.
═══════════════════════════════════════════════════════════════════════════════
What it does:
  - Loads your trained model (best_model.pt)
  - Runs detection on the image
  - Draws bounding boxes with class name + confidence %
  - Shows severity level and first aid advice
  - Saves annotated image to runs/inference/
  - Saves detection results as JSON
═══════════════════════════════════════════════════════════════════════════════
Usage:
  python3 inference.py --source image.jpg
  python3 inference.py --source images/              (whole folder)
  python3 inference.py --source image.jpg --conf 0.25   (sensitive)
  python3 inference.py --source image.jpg --conf 0.50   (conservative)
  python3 inference.py --source image.jpg --weights path/to/best.pt
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import sys
from pathlib import Path

from utilities.data_utils import (
    CLASS_NAMES, find_best_weights,
    print_header, print_success, print_warn, print_error, print_info,
)


"""
═══════════════════════════════════════════════════════════════════════════════
CONFIGURATION
═══════════════════════════════════════════════════════════════════════════════
"""

# Colors in BGR format (OpenCV uses BGR not RGB)
CLASS_COLORS_BGR = [
    (80,  80,  255),   # background — red
    (0,   165, 255),   # normal     — orange
    (50,  220, 255),   # diabetic   — yellow
    (100, 255, 100),   # pressure   — green
    (180, 0,   180),   # surgical   — purple
    (255, 180, 0  ),   # venous     — cyan
]

SEVERITY = {
    "background": ("None",     "No wound detected"),
    "normal":     ("None",     "Healthy skin"),
    "diabetic":   ("High",     "Seek immediate medical care"),
    "pressure":   ("High",     "Relieve pressure immediately"),
    "surgical":   ("Moderate", "Follow surgeon's instructions"),
    "venous":     ("High",     "Elevate and apply compression"),
}

FIRST_AID = {
    "background": "No wound detected in this region.",
    "normal":     "Healthy skin — no treatment needed.",
    "diabetic":   "Clean gently. Do NOT self-treat. Seek immediate medical care.",
    "pressure":   "Relieve pressure immediately. Keep clean. See wound care specialist.",
    "surgical":   "Keep sterile. Do not remove dressing. Follow surgeon's instructions.",
    "venous":     "Elevate leg. Apply compression bandage. Seek vascular specialist.",
}


"""
═══════════════════════════════════════════════════════════════════════════════
ARGUMENTS
═══════════════════════════════════════════════════════════════════════════════
"""

def parse_args():
    p = argparse.ArgumentParser(description="Run wound detection on an image")
    p.add_argument("--source",  required=True,
                   help="Image path or folder of images")
    p.add_argument("--conf",    type=float, default=0.25,
                   help="Confidence threshold (default: 0.25)")
    p.add_argument("--iou",     type=float, default=0.45,
                   help="IoU threshold (default: 0.45)")
    p.add_argument("--weights", type=str, default=None,
                   help="Path to specific weights (default: best available)")
    return p.parse_args()


"""
═══════════════════════════════════════════════════════════════════════════════
ANNOTATION
═══════════════════════════════════════════════════════════════════════════════
"""

def annotate_image(frame, boxes, conf_thresh):
    """
    Draw bounding boxes, labels and severity info on the image.
    Returns (annotated_frame, list_of_detections).
    """
    import cv2

    annotated  = frame.copy()
    detections = []

    for box in boxes:
        conf = float(box.conf[0])
        cls  = int(box.cls[0])

        if conf < conf_thresh or cls >= len(CLASS_NAMES):
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color    = CLASS_COLORS_BGR[cls]
        cls_name = CLASS_NAMES[cls]
        label    = f"{cls_name}  {conf:.0%}"

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label background + text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
        cv2.putText(annotated, label, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        detections.append({
            "class":      cls_name,
            "confidence": round(conf, 4),
            "severity":   SEVERITY.get(cls_name, ("Unknown", ""))[0],
            "first_aid":  FIRST_AID.get(cls_name, "Seek medical attention."),
            "bbox":       [x1, y1, x2, y2],
        })

    # Detection count banner
    if detections:
        cv2.putText(annotated, f"Detections: {len(detections)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 220, 220), 2)

    return annotated, detections


"""
═══════════════════════════════════════════════════════════════════════════════
PROCESS IMAGE
═══════════════════════════════════════════════════════════════════════════════
"""

def process_image(model, img_path, args, output_dir):
    """Run inference on a single image and save results."""
    import cv2

    frame = cv2.imread(str(img_path))
    if frame is None:
        print_warn(f"Cannot read: {img_path}")
        return []

    results   = model(frame, conf=args.conf, iou=args.iou,
                      imgsz=640, verbose=False)
    annotated, detections = annotate_image(frame, results[0].boxes, args.conf)

    # Save annotated image
    out_path = output_dir / f"detected_{img_path.name}"
    cv2.imwrite(str(out_path), annotated)

    # Print results
    print_info(f"{img_path.name} → {len(detections)} detection(s)")
    for det in detections:
        sev = SEVERITY.get(det["class"], ("Unknown", ""))[0]
        print_info(f"  [{det['class']}]  conf={det['confidence']:.1%}  severity={sev}")
        print_info(f"  First aid: {det['first_aid']}")

    return detections


"""
═══════════════════════════════════════════════════════════════════════════════
MAIN
═══════════════════════════════════════════════════════════════════════════════
"""

def inference():
    args = parse_args()

    print_header("Inference — Wound Detection")

    # Load weights
    weights = args.weights if args.weights else find_best_weights()
    if not weights:
        print_error("No trained weights found. Run train.py first.")
        sys.exit(1)

    source = Path(args.source)
    if not source.exists():
        print_error(f"Source not found: {args.source}")
        print_info("Upload the image to wound_system/ first.")
        sys.exit(1)

    print_info(f"Weights   : {weights}")
    print_info(f"Source    : {args.source}")
    print_info(f"Threshold : {args.conf}\n")

    from ultralytics import YOLO
    model = YOLO(weights)

    output_dir = Path("runs/inference")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_detections = []

    if source.is_dir():
        images = (list(source.glob("*.jpg")) +
                  list(source.glob("*.jpeg")) +
                  list(source.glob("*.png")))
        print_info(f"Found {len(images)} images in folder\n")
        for img in images:
            dets = process_image(model, img, args, output_dir)
            all_detections.extend(dets)
    else:
        all_detections = process_image(model, source, args, output_dir)

    # Save JSON results
    if all_detections:
        json_path = output_dir / "detections.json"
        with open(json_path, "w") as f:
            json.dump(all_detections, f, indent=2)
        print_success(f"JSON results saved → {json_path}")

    print_success(f"Annotated images saved → {output_dir}/")


if __name__ == "__main__":
    inference()
