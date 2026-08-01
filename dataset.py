"""
═══════════════════════════════════════════════════════════════════════════════
DATASET PREPARATION
═══════════════════════════════════════════════════════════════════════════════
Handles everything related to dataset preparation in one file:
    1. DOWNLOAD - Clones AZH clinical wound dataset from GitHub
    2. CONVERT  - Converts to YOLO format (uses both Train + Test folders)
    3. VALIDATE - Checks dataset structure, labels, and class distribution
═══════════════════════════════════════════════════════════════════════════════
Usage:
    python3 dataset.py                     (runs all 3 steps in order)
    python3 dataset.py --step download     (download only)
    python3 dataset.py --step convert      (convert only)
    python3 dataset.py --step validate     (validate only)
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

"""
═══════════════════════════════════════════════════════════════════════════════
CONFIGURATION
═══════════════════════════════════════════════════════════════════════════════
"""

AZH_REPO_URL     = "https://github.com/uwm-bigdata/wound-classification-using-images-and-locations.git"
AZH_FOLDER       = "azh_dataset"
AZH_TRAIN_FOLDER = "azh_dataset/dataset/Train"
AZH_TEST_FOLDER  = "azh_dataset/dataset/Test"
OUTPUT_FOLDER    = "wound_dataset"
DATASET_YAML     = "dataset.yaml"
RANDOM_SEED      = 42
BOX_COVERAGE     = 1.0

TRAIN_RATIO    = 0.70
VALIDATE_RATIO = 0.20
TEST_RATIO     = 0.10

CLASS_NAMES = [
    "background",   # 0 - folder: BG
    "normal",       # 1 - folder: N
    "diabetic",     # 2 - folder: D
    "pressure",     # 3 - folder: P
    "surgical",     # 4 - folder: S
    "venous",       # 5 - folder: V
]

CLASS_LABELS = {
    "background": "Background / Non-Wound",
    "normal":     "Normal Healthy Skin",
    "diabetic":   "Diabetic Ulcer",
    "pressure":   "Pressure Wound",
    "surgical":   "Surgical / Post-Operative Wound",
    "venous":     "Venous Ulcer",
}

CLASS_FOLDER_HINTS = {
    "background": ["BG", "bg", "Bg", "background"],
    "normal":     ["N",  "normal",   "Normal"],
    "diabetic":   ["D",  "diabetic", "Diabetic"],
    "pressure":   ["P",  "pressure", "Pressure"],
    "surgical":   ["S",  "surgical", "Surgical"],
    "venous":     ["V",  "venous",   "Venous"],
}


# ═════════════════════════════════════════════════════════════════════════════
# PRINT HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def header(title):
    print(f"\n{'═' * 55}")
    print(f"   {title}")
    print(f"{'═' * 55}")

def success(msg): print(f"  ✅ {msg}")
def warn(msg):    print(f"  ⚠️  {msg}")
def error(msg):   print(f"  ❌ {msg}")
def info(msg):    print(f"     {msg}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — DOWNLOAD
# ═════════════════════════════════════════════════════════════════════════════

def download():
    header("STEP 1 — Download AZH Clinical Wound Dataset")

    info("Source : github.com/uwm-bigdata/wound-classification-using-images-and-locations")
    info("Images : 930 clinical wound images")
    info("Labels : Annotated by wound specialist\n")

    if Path(AZH_FOLDER).exists():
        imgs = (list(Path(AZH_FOLDER).rglob("*.jpg")) +
                list(Path(AZH_FOLDER).rglob("*.png")))
        if imgs:
            warn(f"Already downloaded ({len(imgs)} images) — skipping.")
            info(f"Delete '{AZH_FOLDER}/' to re-download.")
            return
        shutil.rmtree(AZH_FOLDER)

    info("Cloning from GitHub (1-2 minutes)...")
    result = subprocess.run(
        ["git", "clone", "--depth=1", AZH_REPO_URL, AZH_FOLDER],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        error("Git clone failed:")
        print(result.stderr)
        info("Make sure you have internet access.")
        sys.exit(1)

    dataset_path = Path(AZH_FOLDER) / "dataset"
    for zip_name in ["Train.zip", "Test.zip"]:
        zip_path = dataset_path / zip_name
        if zip_path.exists():
            info(f"Unzipping {zip_name}...")
            subprocess.run(
                ["unzip", "-q", str(zip_path), "-d", str(dataset_path)],
                capture_output=True
            )

    imgs = (list(Path(AZH_FOLDER).rglob("*.jpg")) +
            list(Path(AZH_FOLDER).rglob("*.png")))
    success(f"Downloaded — {len(imgs)} images found")

    train_path = dataset_path / "Train"
    if train_path.exists():
        info("\nAvailable class folders:")
        for folder in sorted(train_path.iterdir()):
            if folder.is_dir():
                n = (len(list(folder.glob("*.jpg"))) +
                     len(list(folder.glob("*.png"))))
                if n > 0:
                    info(f"  {folder.name}/ — {n} images")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — CONVERT
# ═════════════════════════════════════════════════════════════════════════════

def _find_class_folder(base_path: Path, class_name: str):
    """Find the folder for a class by trying all name hints."""
    hints = CLASS_FOLDER_HINTS.get(class_name, [class_name])
    for folder in base_path.rglob("*"):
        if folder.is_dir():
            for hint in hints:
                if hint.lower() == folder.name.lower():
                    imgs = (list(folder.glob("*.jpg")) +
                            list(folder.glob("*.jpeg")) +
                            list(folder.glob("*.png")))
                    if imgs:
                        return folder, imgs
    return None, []


def _create_yolo_label(label_path: Path, class_id: int):
    """
    Write a YOLO format label with a centered bounding box.
    AZH images are pre-cropped to the wound region so a
    centered 90% box is accurate. No manual drawing needed.

    YOLO format: class_id  x_center  y_center  width  height
    All values normalised between 0 and 1.
    """
    label_path.write_text(
        f"{class_id} 0.500000 0.500000 "
        f"{BOX_COVERAGE:.6f} {BOX_COVERAGE:.6f}\n"
    )


def _write_dataset_yaml():
    """Write dataset.yaml configuration file for YOLO26."""
    names_str = "\n".join(
        f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)
    )
    Path(DATASET_YAML).write_text(f"""# dataset.yaml
# Auto-generated by dataset.py
# Tells YOLO26 where your data is and what the classes are.

path: ./{OUTPUT_FOLDER}
train: train/images
val:   val/images
test:  test/images

nc: {len(CLASS_NAMES)}
names:
{names_str}
""")
    success(f"dataset.yaml written — {len(CLASS_NAMES)} classes")


def convert():
    header("STEP 2 — Convert to YOLO Format")

    info(f"Source Train : {AZH_TRAIN_FOLDER}")
    info(f"Source Test  : {AZH_TEST_FOLDER}")
    info(f"Output       : {OUTPUT_FOLDER}/")
    info(f"Box coverage : {BOX_COVERAGE:.0%} — auto-generated, no manual drawing\n")

    azh_train_path = Path(AZH_TRAIN_FOLDER)
    azh_test_path  = Path(AZH_TEST_FOLDER)

    if not azh_train_path.exists():
        error(f"Source not found: {AZH_TRAIN_FOLDER}")
        info("Run --step download first.")
        sys.exit(1)

    # Create output folder structure
    for split in ["train", "val", "test"]:
        Path(f"{OUTPUT_FOLDER}/{split}/images").mkdir(parents=True, exist_ok=True)
        Path(f"{OUTPUT_FOLDER}/{split}/labels").mkdir(parents=True, exist_ok=True)
    success(f"Output folders created -> {OUTPUT_FOLDER}/")

    random.seed(RANDOM_SEED)
    total = 0

    for class_id, class_name in enumerate(CLASS_NAMES):
        info(f"Processing class {class_id}: {class_name}")

        # Get images from Train folder
        _, images_train = _find_class_folder(azh_train_path, class_name)

        # Get images from Test folder (if exists)
        images_test = []
        if azh_test_path.exists():
            _, images_test = _find_class_folder(azh_test_path, class_name)

        # Combine both Train and Test images
        images = images_train + images_test

        if not images:
            warn(f"  No images found for '{class_name}' — skipping")
            continue

        info(f"  Found {len(images_train)} train + {len(images_test)} test = {len(images)} total images")

        # Shuffle and split 70 / 20 / 10
        random.shuffle(images)
        n       = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VALIDATE_RATIO)

        splits = {
            "train": images[:n_train],
            "val":   images[n_train:n_train + n_val],
            "test":  images[n_train + n_val:],
        }

        for split_name, split_images in splits.items():
            for img_path in split_images:
                new_name = f"{class_name}_{img_path.name}"

                shutil.copy2(
                    img_path,
                    Path(f"{OUTPUT_FOLDER}/{split_name}/images/{new_name}")
                )

                _create_yolo_label(
                    Path(f"{OUTPUT_FOLDER}/{split_name}/labels/"
                         f"{class_name}_{img_path.stem}.txt"),
                    class_id
                )
                total += 1

        success(
            f"  {class_name}: {len(images)} images -> "
            f"{int(n*TRAIN_RATIO)} train / "
            f"{int(n*VALIDATE_RATIO)} val / "
            f"{n - int(n*TRAIN_RATIO) - int(n*VALIDATE_RATIO)} test"
        )

    info(f"\nTotal images converted: {total}")

    if total == 0:
        error("No images were converted!")
        info("Check that azh_dataset/dataset/Train/ has BG/N/D/P/S/V folders.")
        sys.exit(1)

    _write_dataset_yaml()
    success("Conversion complete!")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — VALIDATE
# ═════════════════════════════════════════════════════════════════════════════

def validate():
    header("STEP 3 — Validate Dataset")

    dataset_path = Path(OUTPUT_FOLDER)
    if not dataset_path.exists():
        error(f"Dataset not found: {OUTPUT_FOLDER}")
        info("Run --step convert first.")
        sys.exit(1)

    all_ok = True

    print(f"\n  {'Split':<8} {'Images':>8} {'Labels':>8} "
          f"{'Missing':>8} {'Orphan':>8}")
    print("  " + "-" * 44)

    for split in ["train", "val", "test"]:
        img_dir = dataset_path / split / "images"
        lbl_dir = dataset_path / split / "labels"

        if not img_dir.exists():
            warn(f"Missing split directory: {img_dir}")
            all_ok = False
            continue

        images  = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        labels  = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []
        i_stems = {p.stem for p in images}
        l_stems = {p.stem for p in labels}
        missing = i_stems - l_stems
        orphans = l_stems - i_stems

        status = "OK" if not missing and not orphans else "!!"
        print(f"  {status} {split:<6} {len(images):>8} {len(labels):>8} "
              f"{len(missing):>8} {len(orphans):>8}")

        if missing or orphans:
            all_ok = False

    print()
    label_errors = 0
    for lbl_path in dataset_path.rglob("*.txt"):
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) != 5:
                    label_errors += 1
                    continue
                if int(parts[0]) >= len(CLASS_NAMES):
                    label_errors += 1

    if label_errors:
        warn(f"{label_errors} label format errors found")
        all_ok = False
    else:
        success("All label files are valid YOLO format")

    print(f"\n  {'Class':<15} {'Train':>8} {'Val':>8} "
          f"{'Test':>8} {'Total':>8}")
    print("  " + "-" * 44)

    for class_id, class_name in enumerate(CLASS_NAMES):
        counts = {}
        for split in ["train", "val", "test"]:
            lbl_dir = dataset_path / split / "labels"
            count = 0
            if lbl_dir.exists():
                for lbl in lbl_dir.glob("*.txt"):
                    with open(lbl) as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts and int(parts[0]) == class_id:
                                count += 1
            counts[split] = count

        total = sum(counts.values())
        status = "OK" if total > 0 else "!!"
        print(f"  {status} {class_name:<13} "
              f"{counts['train']:>8} {counts['val']:>8} "
              f"{counts['test']:>8} {total:>8}")

    lbl_dir = dataset_path / "train" / "labels"
    if lbl_dir.exists():
        cls_counts = defaultdict(int)
        for lbl in lbl_dir.glob("*.txt"):
            with open(lbl) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_counts[int(parts[0])] += 1

        if cls_counts:
            max_c = max(cls_counts.values())
            min_c = min(cls_counts.values())
            ratio = max_c / max(min_c, 1)
            print()
            if ratio > 3:
                warn(f"Class imbalance detected — ratio: {ratio:.1f}x")
                info("Add more images for smaller classes to improve recall.")
            else:
                success(f"Class balance is okay (ratio: {ratio:.1f}x)")

    print()
    if all_ok:
        success("Dataset is valid — ready for training!")
    else:
        warn("Some issues found. Training may still work but review above.")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="WoundSight Data Preparation — download, convert, validate",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--step", default="all",
                   choices=["all", "download", "convert", "validate"],
                   help=(
                       "Which step to run (default: all):\n"
                       "  all      -> download + convert + validate\n"
                       "  download -> clone AZH dataset from GitHub\n"
                       "  convert  -> convert to YOLO format\n"
                       "  validate -> check dataset structure\n"
                   ))
    return p.parse_args()


def main():
    args = parse_args()

    print("\n WoundSight — Data Preparation")
    print(f"   Running: {args.step}\n")

    if args.step in ("download", "all"):
        download()

    if args.step in ("convert", "all"):
        convert()

    if args.step in ("validate", "all"):
        validate()

    print(f"\n{'=' * 55}")
    print(f"  dataset.py --step {args.step} complete!")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
