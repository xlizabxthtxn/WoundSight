"""
═══════════════════════════════════════════════════════════════════════════════
MODEL TRAINING
═══════════════════════════════════════════════════════════════════════════════
Trains the YOLO26m wound detection model.
═══════════════════════════════════════════════════════════════════════════════
What it does:
  1. Checks GPU is available
  2. Loads YOLO26m and YOLO26l pretrained weights 
  3. Trains on your wound dataset 
  4. Saves best weights to best_model.pt when done
  5. Displays F1 score after every epoch
═══════════════════════════════════════════════════════════════════════════════
Usage:
  python3 train.py                          (default settings)
  python3 train.py --model yolo26l.pt       (bigger model)
  python3 train.py --batch 8                (less GPU memory)
  python3 train.py --device cpu             (CPU only, slow)
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import shutil
import sys
from pathlib import Path

from utilities.data_utils import (
    print_header, print_success, print_warn, print_error, print_info,
    find_best_weights,
)

"""
═══════════════════════════════════════════════════════════════════════════════
Default Configuration
═══════════════════════════════════════════════════════════════════════════════
"""
DATASET_YAML   = "dataset.yaml"
DEFAULT_MODEL  = "yolo26m.pt"
DEFAULT_BATCH  = 16
DEFAULT_DEVICE = "0"      # "0" = GPU, "cpu" = CPU
WEIGHTS_BACKUP = "best_model.pt"


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO26 Wound Detection Model")
    p.add_argument("--model",   default=DEFAULT_MODEL,
                   help=f"Model size (default: {DEFAULT_MODEL})")
    p.add_argument("--epochs",  type=int, required=True,
                   help="Number of epochs e.g. --epochs 150")
    p.add_argument("--batch",   type=int, default=DEFAULT_BATCH,
                   help=f"Batch size (default: {DEFAULT_BATCH})")
    p.add_argument("--device",  default=DEFAULT_DEVICE,
                   help="'0' for GPU, 'cpu' for CPU")
    p.add_argument("--name", default="wound_detection",
                   help="Run name e.g. --name wound_ep50")
    return p.parse_args()


def check_gpu(device):
    """Check if GPU is available when device='0'."""
    if device == "cpu":
        print_warn("Running on CPU — Training will be slow.")
        return "cpu"

    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            print_success(f"GPU detected: {gpu}")
            return "0"
        else:
            print_warn("GPU not detected — Falling back to CPU.")
            print_info("Fix: pip install torch==2.4.0 torchvision==0.19.0 "
                       "--index-url https://download.pytorch.org/whl/cu124")
            return "cpu"
    except Exception:
        print_warn("Could not check GPU — Using CPU.")
        return "cpu"


def on_val_end(trainer):
    """Print F1 score after every validation epoch."""
    try:
        p     = trainer.validator.metrics.box.mp
        r     = trainer.validator.metrics.box.mr
        map50 = trainer.validator.metrics.box.map50
        f1    = 2 * p * r / (p + r) if (p + r) > 0 else 0
        ep    = trainer.epoch + 1
        total = trainer.epochs
        f1_str = f"{f1:.4f}"
        if f1 >= 0.85:
            f1_str = f"✅ {f1_str}"
        elif f1 >= 0.80:
            f1_str = f"🟡 {f1_str}"
        else:
            f1_str = f"🔴 {f1_str}"
        print(f"  │ Epoch {ep:>3}/{total}  "
              f"F1: {f1_str}  "
              f"mAP50: {map50:.4f}  "
              f"P: {p:.4f}  R: {r:.4f}")
    except Exception:
        pass  # silently skip if metrics not ready yet


def train():
    args = parse_args()

    print_header("Train — YOLO26m Wound Detection")
    print_info(f"Model    : {args.model}")
    print_info(f"Epochs   : {args.epochs}")
    print_info(f"Batch    : {args.batch}")
    print_info(f"Dataset  : {DATASET_YAML}\n")

    if not Path(DATASET_YAML).exists():
        print_error("dataset.yaml not found. Run convert.py first.")
        sys.exit(1)

    # Check GPU
    device = check_gpu(args.device)
    est = "1–2 hours" if device == "cpu" else "30–60 minutes"
    print_info(f"Estimated time: {est}\n")
    print_info("Keep this terminal open — do not close VSCode!\n")

    try:
        from ultralytics import YOLO

        print_info(f"Loading {args.model}...")
        model = YOLO(args.model)

        # ── Register F1 callback ──────────────────────────────────────────────
        model.add_callback("on_val_end", on_val_end)

        print_info("Starting training...\n")
        print_info("F1 will be shown after each epoch:")
        print_info("  ✅ F1 ≥ 0.85 (target met)")
        print_info("  🟡 F1 ≥ 0.80 (getting close)")
        print_info("  🔴 F1 < 0.80 (still learning)\n")

        results = model.train(
            data         = DATASET_YAML,
            name         = args.name,
            epochs       = args.epochs,
            batch        = args.batch,
            imgsz        = 640,
            device       = device,
            project      = "runs",
            patience     = 0,        
            save         = True,
            save_period  = 10,        
            exist_ok     = True,
            optimizer    = "SGD",
            lr0          = 0.01,
            lrf          = 0.01,
            momentum     = 0.937,
            weight_decay = 0.0005,
            warmup_epochs= 3,
            box          = 7.5,
            cls          = 0.5,
            hsv_h        = 0.015,
            hsv_s        = 0.7,
            hsv_v        = 0.4,
            degrees      = 10.0,
            translate    = 0.1,
            scale        = 0.5,
            flipud       = 0.0,
            fliplr       = 0.5,
            mosaic       = 1.0,
            mixup        = 0.1,
            verbose      = True,
            seed         = 42,
        )

        # Backup best weights to project root
        best_pt = Path(results.save_dir) / "weights" / "best.pt"
        if best_pt.exists():
            shutil.copy2(best_pt, WEIGHTS_BACKUP)
            print()
            print_success("Training complete!")
            print_success(f"Best weights saved → {best_pt}")
            print_success(f"Backup created    → {WEIGHTS_BACKUP}")
            print()
            print_info("Next steps:")
            print_info("  python3 metrics.py    (see your results)")
            print_info("  python3 evaluate.py   (test set evaluation)")
            print_info("  python3 inference.py  (test on an image)")
        else:
            print_warn("Training finished but best.pt not found.")

    except Exception as e:
        print_error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    train()