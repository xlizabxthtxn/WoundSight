# WoundSight
**A Deep Learning-Based Automated Wound Detection and Classification System**

Capstone Project 2 | Elizabeth Tan Wei | ID: 22038970 | Sunway University | 2026

## Overview
WoundSight uses YOLO26m trained on the AZH Clinical Wound Dataset (930 images, 6 classes)
to detect and classify chronic wounds in real time via a Flask web application.

## Best Result
SGD optimiser, 250 epochs — F1: 0.850, mAP50: 0.878

## Classes
Background, Normal Skin, Diabetic Ulcer, Pressure Wound, Surgical Wound, Venous Ulcer

## Setup
```bash
pip install ultralytics flask werkzeug opencv-python matplotlib pandas
```

## Run App
```bash
python3 app.py
```

## Train
```bash
python3 train.py --device 0 --epochs 250 --name SGD_250
```

## Hardware
NVIDIA L4 GPU, AWS ParallelCluster, Python 3.12, PyTorch 2.4.0+cu124

## Dataset
AZH Clinical Wound Dataset — 930 images, 6 classes
- Source: AZH Wound and Vascular Center, Milwaukee, Wisconsin
- Split: 70:20:10 (train/val/test), seed 42
- Classes: Background, Normal, Diabetic, Pressure, Surgical, Venous
