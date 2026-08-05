# WoundSight
### A Deep Learning-Based Automated Wound Detection and Classification System

**Author:** Elizabeth | Student ID: 22038970  
**Institution:** Sunway University  
**Project:** Capstone Project — 2026  

---

## Overview

WoundSight is an AI-powered wound detection and classification system built on the YOLO26m architecture, trained on the AZH Clinical Wound Dataset. The system simultaneously localises and classifies chronic wounds from a single uploaded image, providing clinical recommendations for both healthcare professionals and patients.

The project follows a two-phase experimental design:
- **Phase 1** — Model size comparison: YOLO26n (nano), YOLO26m (medium), YOLO26l (large)
- **Phase 2** — Optimiser comparison: AdamW vs SGD across 50–250 epochs

**Best result:** SGD optimiser, 250 epochs → F1: 0.850, mAP50: 0.878

---

## Wound Classes

| Class | Description |
|---|---|
| Background | Non-wound region |
| Normal Skin | Healthy skin |
| Diabetic Ulcer | Diabetic foot/leg ulcer |
| Pressure Wound | Pressure injury / bedsore |
| Surgical Wound | Post-operative wound |
| Venous Ulcer | Venous leg ulcer |

---

## Dataset

**AZH Clinical Wound Dataset**
- Source: AZH Wound and Vascular Center, Milwaukee, Wisconsin
- Total: 930 images across 6 classes
- Split: 70:20:10 (Train: 648 / Val: 184 / Test: 98), seed 42

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| Background | 70 | 20 | 10 | 100 |
| Normal Skin | 70 | 20 | 10 | 100 |
| Diabetic | 129 | 37 | 19 | 185 |
| Pressure | 93 | 26 | 15 | 134 |
| Surgical | 114 | 32 | 18 | 164 |
| Venous | 172 | 49 | 26 | 247 |

---

## Requirements

### Hardware
- NVIDIA GPU (tested on NVIDIA L4, 23GB VRAM)
- Minimum 16GB RAM recommended

### Software
```bash
Python 3.12
PyTorch 2.4.0+cu124
```

### Install dependencies
```bash
pip install ultralytics flask werkzeug opencv-python matplotlib pandas albumentations
```

---

## Project Structure

```
WoundSight/
├── app.py              # Flask web application
├── train.py            # Model training script
├── dataset.py          # Dataset preparation and splitting
├── evaluation.py       # Model evaluation on test set
├── metrics.py          # Metrics computation and summary
├── inference.py        # CLI inference on single image or folder
├── inference_time.py   # Inference time measurement
├── train_time.py       # Training time logging
├── dataset.yaml        # Dataset configuration for YOLO
├── utilities/          # Helper functions and utilities
├── runs/               # Training outputs (results, curves, weights)
├── metrics/            # Per-run metric summaries and curves
├── wound_dataset/      # Processed dataset (YOLO format)
└── azh_dataset/        # Original AZH dataset
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/xlizabxthtxn/WoundSight.git
cd WoundSight
```

### 2. Create a virtual environment
```bash
python3 -m venv Wound_Environment --system-site-packages
source Wound_Environment/bin/activate
```

### 3. Install dependencies
```bash
pip install ultralytics flask werkzeug opencv-python matplotlib pandas
```

### 4. Download pretrained YOLO weights
```bash
# These will auto-download on first training run, or manually:
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo26n.pt
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo26m.pt
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo26l.pt
```

---

## Usage

### 1. Prepare the Dataset
```bash
python3 dataset.py
```
Splits the AZH dataset into train/val/test sets and generates YOLO-format bounding box annotations.

---

### 2. Train a Model

**Basic training (SGD, default settings):**
```bash
python3 train.py --device 0 --epochs 250 --name SGD_250
```

**Training options:**
```bash
python3 train.py --help

  --model    yolo26m.pt       # Model size: yolo26n.pt / yolo26m.pt / yolo26l.pt
  --epochs   250              # Number of training epochs
  --batch    16               # Batch size
  --device   0                # GPU ID (0, 1, 2, 3) or cpu
  --name     SGD_250          # Run name (saved to runs/detect/runs/)
```

**Phase 1 — Model size comparison (run in parallel on separate GPUs):**
```bash
python3 train.py --model yolo26n.pt --epochs 250 --name Nano_SGD_250 --device 0
python3 train.py --model yolo26m.pt --epochs 250 --name SGD_250       --device 1
python3 train.py --model yolo26l.pt --epochs 250 --name Large_SGD_250 --device 2
```

**Phase 2 — Optimiser comparison (change optimizer in train.py):**
```bash
# SGD (default in train.py)
python3 train.py --epochs 50  --name SGD_50  --device 0
python3 train.py --epochs 100 --name SGD_100 --device 1
python3 train.py --epochs 150 --name SGD_150 --device 2
python3 train.py --epochs 200 --name SGD_200 --device 3
python3 train.py --epochs 250 --name SGD_250 --device 0

# AdamW (change optimizer="AdamW", lr0=0.001, momentum=0.9 in train.py)
python3 train.py --epochs 50  --name AdamW_50  --device 0
python3 train.py --epochs 100 --name AdamW_100 --device 1
python3 train.py --epochs 150 --name AdamW_150 --device 2
python3 train.py --epochs 200 --name AdamW_200 --device 3
python3 train.py --epochs 250 --name AdamW_250 --device 0
```

---

### 3. View Metrics

```bash
python3 metrics.py
```
Prints a summary table of all runs showing mAP50, Precision, Recall and F1 for each configuration.

---

### 4. Evaluate on Test Set

```bash
python3 evaluation.py --weights runs/detect/runs/SGD_250/weights/best.pt --split test
```

**Evaluate all runs:**
```bash
python3 evaluation.py --all
```

---

### 5. Run Inference on an Image

```bash
# Basic inference (uses best available model)
python3 inference.py --source image.jpg

# Specify model weights
python3 inference.py --source image.jpg \
  --weights runs/detect/runs/SGD_250/weights/best.pt

# Run on a folder of images
python3 inference.py --source wound_dataset/test/images/

# Adjust confidence threshold
python3 inference.py --source image.jpg --conf 0.15
```

Output saved to `runs/inference/` as annotated images and a JSON results file.

---

### 6. Run the Web Application

```bash
# Kill any existing process on port 5000
fuser -k 5000/tcp

# Start the app
python3 app.py
```

Then open your browser at:
```
http://localhost:5000
```

**App features:**
- Upload a wound image (JPG, PNG)
- Primary detection using SGD_250 (best model, F1 0.850)
- Side-by-side comparison: AdamW 250ep vs SGD 250ep
- Clinical recommendations (doctor and patient versions)
- Confidence tier notes (High / Moderate / Low)
- Metrics dashboard with Phase 1 and Phase 2 results
- Dataset information and per-class distribution

---

### 7. Measure Inference Time

```bash
python3 inference_time.py
```

Reports average inference time per image (ms) and FPS for each trained model.

---

### 8. Measure Training Time

```bash
python3 train_time.py
```

Reports training time per epoch and total training time for each run.

---

## Results

### Phase 1 — Model Size Comparison (SGD, 250 Epochs)

| Model | Params | mAP50 | Precision | Recall | F1 |
|---|---|---|---|---|---|
| YOLO26n (Nano) | ~3M | — | — | — | — |
| YOLO26m (Medium) | 20.3M | 0.878 | 0.854 | 0.846 | **0.850** |
| YOLO26l (Large) | ~43M | — | — | — | — |

### Phase 2 — Optimiser Comparison (YOLO26m)

| Run | Optimiser | Epochs | mAP50 | F1 |
|---|---|---|---|---|
| AdamW_50 | AdamW | 50 | 0.829 | 0.760 |
| AdamW_100 | AdamW | 100 | 0.816 | 0.758 |
| AdamW_150 | AdamW | 150 | 0.855 | 0.786 |
| AdamW_200 | AdamW | 200 | 0.877 | 0.813 |
| AdamW_250 | AdamW | 250 | 0.895 | 0.834 |
| SGD_50 | SGD | 50 | 0.825 | 0.775 |
| SGD_100 | SGD | 100 | 0.878 | 0.799 |
| SGD_150 | SGD | 150 | 0.881 | 0.828 |
| SGD_200 | SGD | 200 | 0.891 | 0.833 |
| **SGD_250** | **SGD** | **250** | **0.878** | **0.850** ✅ |

**Supervisor target: F1 ≥ 0.85 — met by SGD_250**

### Per-Class Results (SGD_250, Validation Set)

| Class | Precision | Recall | F1 | mAP50 |
|---|---|---|---|---|
| Background | 0.959 | 0.950 | 0.954 | 0.990 |
| Normal Skin | 0.945 | 0.950 | 0.947 | 0.977 |
| Diabetic | 0.753 | 0.757 | 0.755 | 0.768 |
| Pressure | 0.741 | 0.615 | 0.672 | 0.690 |
| Surgical | 0.883 | 0.812 | 0.846 | 0.889 |
| Venous | 0.761 | 0.939 | 0.841 | 0.941 |

---

## Hardware Used

```
GPU:      NVIDIA L4 (23GB VRAM)
Platform: AWS ParallelCluster via Open OnDemand
Python:   3.12
PyTorch:  2.4.0+cu124
CUDA:     12.4
```

---

## Disclaimer

This system is intended for research and educational purposes only. Results should not be used as a substitute for professional medical diagnosis. Always consult a qualified healthcare professional for wound assessment and treatment.

---

## License

This project was developed as part of a Capstone Project at Sunway University. The AZH Clinical Wound Dataset is used for research purposes only and remains the property of its original authors.
