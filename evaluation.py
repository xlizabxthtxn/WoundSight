"""
═══════════════════════════════════════════════════════════════════════════════
EVALUATION
═══════════════════════════════════════════════════════════════════════════════
Evaluates the trained model on the test set.
═══════════════════════════════════════════════════════════════════════════════
Why use the test set?
  - Train set : model learns from these 
  - Val set   : used during training to monitor progress
  - Test set  : model has NEVER seen these — gives honest accuracy

This is your official final accuracy number for your report.
═══════════════════════════════════════════════════════════════════════════════
What it outputs:
  - mAP50 and mAP50-95 overall
  - Precision, Recall, F1 per class
  - Confusion matrix (saved as PNG)
═══════════════════════════════════════════════════════════════════════════════
Usage:
  python3 evaluation.py
  python3 evaluation.py --split val       (evaluate on val set instead)
  python3 evaluation.py --conf 0.25       (change confidence threshold)
  python3 evaluation.py --all             (evaluate all training runs)
  python3 evaluation.py --weights path    (evaluate specific weights)
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import sys
from pathlib import Path

from utilities.data_utils import (
    CLASS_NAMES, find_best_weights,
    print_header, print_success, print_error, print_info, print_warn,
)


"""
═══════════════════════════════════════════════════════════════════════════════
CONFIGURATION
═══════════════════════════════════════════════════════════════════════════════
"""

DATASET_YAML = "dataset.yaml"


"""
═══════════════════════════════════════════════════════════════════════════════
ARGUMENTS
═══════════════════════════════════════════════════════════════════════════════
"""

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate wound detection model")
    p.add_argument("--split",   default="test",
                   choices=["train", "val", "test"],
                   help="Which split to evaluate on (default: test)")
    p.add_argument("--conf",    type=float, default=0.25,
                   help="Confidence threshold (default: 0.25)")
    p.add_argument("--iou",     type=float, default=0.50,
                   help="IoU threshold (default: 0.50)")
    p.add_argument("--weights", type=str, default=None,
                   help="Path to specific weights file")
    p.add_argument("--all",     action="store_true",
                   help="Evaluate all training runs")
    return p.parse_args()


"""
═══════════════════════════════════════════════════════════════════════════════
EVALUATE SINGLE MODEL
═══════════════════════════════════════════════════════════════════════════════
"""

def evaluate(weights=None, split="test", conf=0.25, iou=0.50):
    print_header(f"Evaluate — {split.upper()} Set")

    if weights is None:
        weights = find_best_weights()
    if not weights:
        print_error("No trained weights found. Run train.py first.")
        sys.exit(1)

    print_info(f"Weights  : {weights}")
    print_info(f"Split    : {split}")
    print_info(f"Conf     : {conf}")
    print_info(f"Dataset  : {DATASET_YAML}\n")

    try:
        from ultralytics import YOLO

        model   = YOLO(weights)
        results = model.val(
            data    = DATASET_YAML,
            split   = split,
            conf    = conf,
            iou     = iou,
            imgsz   = 640,
            plots   = True,
            verbose = True,
        )

        box     = results.box
        map50   = float(box.map50)
        map5095 = float(box.map)

        # Per-class results table
        print(f"\n  {'─' * 58}")
        print(f"  {'Class':<20} {'Prec':>7} {'Rec':>7} {'F1':>7} {'mAP50':>8}")
        print(f"  {'─' * 58}")

        for i, cls in enumerate(CLASS_NAMES):
            try:
                p  = float(box.p[i])
                r  = float(box.r[i])
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
                ap = float(box.ap50[i])
                print(f"  {cls:<20} {p:>7.3f} {r:>7.3f} {f1:>7.3f} {ap:>8.3f}")
            except (IndexError, AttributeError):
                print(f"  {cls:<20} {'N/A':>7} {'N/A':>7} {'N/A':>7} {'N/A':>8}")

        print(f"  {'─' * 58}")
        print(f"  {'ALL':<20} {'':>7} {'':>7} {'':>7} {map50:>8.3f}")
        print(f"\n  mAP50    : {map50:.4f}  ({map50:.1%})")
        print(f"  mAP50-95 : {map5095:.4f}  ({map5095:.1%})")
        print()
        print_success("Evaluation complete!")
        print_info("Plots saved in runs folder (confusion matrix, PR curve)")

        return {"map50": map50, "map5095": map5095}

    except Exception as e:
        print_error(f"Evaluation failed: {e}")
        sys.exit(1)


"""
═══════════════════════════════════════════════════════════════════════════════
EVALUATE ALL RUNS
═══════════════════════════════════════════════════════════════════════════════
"""

def evaluate_all(split="test", conf=0.25, iou=0.50):
    print_header("Evaluate All — Comparing All Training Runs")

    if not Path("runs").exists():
        print_error("No runs folder found. Run train.py first.")
        sys.exit(1)

    # Find all best.pt files excluding smoketest
    weight_files = [
        p for p in Path("runs").rglob("best.pt")
        if "smoketest" not in str(p)
    ]

    if not weight_files:
        print_error("No trained weights found.")
        sys.exit(1)

    weight_files = sorted(weight_files, key=lambda p: p.stat().st_mtime)
    print_info(f"Found {len(weight_files)} trained model(s)\n")

    all_results = {}

    for weights_path in weight_files:
        run_name = weights_path.parent.parent.name
        print(f"\n  Evaluating: {run_name}")
        print("  " + "-" * 50)
        result = evaluate(
            weights = str(weights_path),
            split   = split,
            conf    = conf,
            iou     = iou,
        )
        all_results[run_name] = result

    # Print comparison summary
    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY — ALL RUNS")
    print("=" * 60)
    print(f"  {'Run':<25} {'mAP50':>8} {'mAP50-95':>10}")
    print("  " + "-" * 45)

    for run_name, result in all_results.items():
        print(f"  {run_name:<25} "
              f"{result['map50']:>8.4f} "
              f"{result['map5095']:>10.4f}")

    best_run = max(all_results, key=lambda k: all_results[k]['map50'])
    print("=" * 60)
    print_success(f"Best model: {best_run} "
                  f"(mAP50: {all_results[best_run]['map50']:.4f})")


"""
═══════════════════════════════════════════════════════════════════════════════
MAIN
═══════════════════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    args = parse_args()

    if args.all:
        evaluate_all(
            split = args.split,
            conf  = args.conf,
            iou   = args.iou,
        )
    else:
        evaluate(
            weights = args.weights,
            split   = args.split,
            conf    = args.conf,
            iou     = args.iou,
        )