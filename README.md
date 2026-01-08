# Object Detection from Scratch

A complete object detection system built from scratch (no pre-training) using PASCAL VOC 2007 dataset.

## Overview

This project implements a custom CNN-based object detector trained from scratch to detect 3 object classes:
- **person**
- **car**
- **dog**

## Project Structure

```
Object Detection/
├── data/
│   ├── voc2007/              # Downloaded PASCAL VOC 2007 dataset
│   └── processed/            # Processed annotations and splits
├── src/
│   ├── dataset.py            # VOC dataset parser and PyTorch Dataset class
│   ├── model.py              # Custom CNN object detector
│   ├── augmentation.py       # Data augmentation with bbox handling
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script (mAP, FPS, model size)
│   └── utils.py              # Helper functions
├── download_voc.py           # Script to download PASCAL VOC 2007
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download PASCAL VOC 2007 dataset:
```bash
python download_voc.py
```

## Usage

### Training

Train the model from scratch:
```bash
python src/train.py
```

Training parameters (can be customized via command-line arguments):
- Learning rate: 1e-3 (default)
- Optimizer: Adam
- Epochs: 30-50 (default: 50)
- Batch size: 8 or 16 (default: 8)
- Loss: Smooth L1 (bbox) + Cross Entropy (classification)
- Input size: 224x224 (default)

Example with custom parameters:
```bash
python src/train.py --batch_size 16 --epochs 30 --lr 0.001
```

### Evaluation

Evaluate the trained model:
```bash
python -m src.evaluate --model_path checkpoints/best_model.pth
```

Metrics computed:
- **mAP@0.5**: Mean Average Precision at IoU ≥ 0.5
- **FPS**: Inference speed (frames per second)
- **Model Size**: File size in MB

Example:
```bash
python -m src.evaluate --model_path checkpoints/best_model.pth --batch_size 8
```

### Real-time Webcam Detection

Run live object detection using your webcam:
```bash
python -m src.webcam_detect --model_path checkpoints/best_model.pth
```

**Controls:**
- Press `q` to quit
- Press `s` to save current frame with detections

**Options:**
```bash
# Use different camera (if you have multiple)
python -m src.webcam_detect --model_path checkpoints/best_model.pth --camera 1

# Adjust detection threshold (lower = more detections, may include false positives)
python -m src.webcam_detect --model_path checkpoints/best_model.pth --score_threshold 0.2

# Display FPS
python -m src.webcam_detect --model_path checkpoints/best_model.pth --fps_display
```

### Image Detection

Detect objects in a single image or directory of images:
```bash
python -m src.inference --model_path checkpoints/best_model.pth --image_path path/to/image.jpg
```

For directory of images:
```bash
python -m src.inference --model_path checkpoints/best_model.pth --image_path path/to/images/
```

## Model Architecture

- **Backbone**: 3 convolution blocks
  - Each block: Conv2d(3×3) → BatchNorm → ReLU → MaxPool2d(2×2)
  - Channels: 3 → 64 → 128 → 256
- **Detection Heads**:
  - Bounding Box Head: Outputs 4 values (x, y, w, h)
  - Classification Head: Outputs 3 class scores + background

## Dataset

- **Source**: PASCAL VOC 2007
- **Classes**: person, car, dog
- **Split**: 70% train, 20% validation, 10% test
- **Augmentation**: Horizontal flip, scaling, brightness, rotation (±10°)

## License

This project is for educational purposes.

