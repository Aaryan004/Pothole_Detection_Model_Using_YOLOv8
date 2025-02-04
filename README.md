# YOLOv8 Pothole Detection 🚀

This repository implements **YOLOv8** to detect potholes in road images. The project involves training a **custom YOLOv8 model** on a dataset formatted in the **YOLO format** and running inference to detect potholes in test images.

## 📌 Table of Contents

- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Prerequisites](#-prerequisites)
- [Setup](#-setup)
- [Usage](#-usage)
  - [Training]
  - [Detection]
  - [GPU Verification]
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

---

## 🚀 Overview

This project uses **YOLOv8** for **pothole detection** in images. The dataset is formatted in the YOLO format with image-label pairs. Training is conducted using Ultralytics' YOLOv8 framework.

**Features:**
- **Custom training** using YOLOv8.
- **Inference and evaluation** on test images.
- **Uses GPU acceleration** for fast training and detection.
- **Real-time performance monitoring** using TensorBoard.

---

## 📂 Repository Structure

  ```bash
  YOLOv8-Pothole-Detection/
  ├── README.md                 # Project documentation
  ├── run_yolov8.py             # Main script for training & detection
  ├── pothole_dataset.yaml      # Dataset configuration file for YOLOv8
  ├── yolo_formatted_data/      # Dataset folder
  │   ├── images/               # Image data
  │   │   ├── train/            # Training images
  │   │   ├── val/              # Validation images
  │   │   ├── test/             # Test images
  │   ├── labels/               # YOLO-format annotations
  │   │   ├── train/            # Training labels
  │   │   ├── val/              # Validation labels
  │   │   ├── test/             # Test labels
  ├── yolov8/                   # YOLOv8 repository from Ultralytics
  │   ├── train.py              # Training script
  │   ├── detect.py             # Detection script
  │   ├── cfg/                  # Model configuration files
  │   ├── models/               # Model architecture files
  │   └── ...                   # Other YOLOv8-related files
  ```
---

## ✅ Prerequisites

- **Python 3.8+**
- **PyTorch (with CUDA support)**  
- **Ultralytics YOLOv8 package**
- **A dataset formatted in YOLO format**
- **NVIDIA GPU with CUDA installed** (Recommended for fast training)

---

## ⚙️ Setup

### 1️⃣ Install YOLOv8

Install Ultralytics YOLOv8 package:
```bash
pip install ultralytics
```
2️⃣ Install PyTorch with CUDA
For CUDA 11+ (Recommended):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
For CPU only (Slower training):
```bash
pip install torch torchvision torchaudio
```
3️⃣ Clone the YOLOv8 Repository (Optional)
```bash
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
pip install -r requirements.txt
```
4️⃣ Configure the Dataset
Update pothole_dataset.yaml:
```bash
path: C:/Users/Aaryan Naithani/OneDrive/Desktop/Pothole_Detection_Model_Using_YOLOv8-main/yolo_formatted_data

train: images/train
val: images/val
test: images/test

nc: 1
names: ['pothole']
```
---
## 🔥 Usage
### 🏋️‍♂️ Training
To train the YOLOv8 model, run:
```bash
python run_yolov8.py
```
Training Parameters:

--epochs 1800 → Number of training epochs.

--batch 16 → Batch size.

--imgsz 640 → Input image size.

--device cuda → Use GPU for training.


###🕵️‍♂️ Detection
Once training is complete, the script will automatically run inference on test images:

```bash

python run_yolov8.py --detect
```
Detection results (bounding boxes) will be saved in:
```bash
runs/detect/pothole_detector_inference/
```

Adjustable Parameters in run_yolov8.py:

--conf 0.25 → Confidence threshold.

--iou 0.7 → IoU threshold for NMS.


### 🖥️ GPU Verification
To verify if YOLOv8 is using the GPU, run:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```
Expected output:

```graphql
True
```
Or, monitor GPU usage with:
```bash
nvidia-smi -l 2
```
---

## 🛠️ Troubleshooting

❌ No Bounding Boxes in Inference Results

🔹 Solution: Try lowering --conf in detection:

```bash
python run_yolov8.py --detect --conf 0.0015
```
❌ CUDA Out of Memory (OOM) Error

🔹 Solutions:

Reduce --batch size.

Lower --imgsz from 640 to 416.

Use --device cpu (slower but avoids OOM).

❌ Dataset Not Found Error

🔹 Solution:

Ensure dataset paths in pothole_dataset.yaml are absolute.

Check that images/train, images/val, and labels/train exist.

❌ PyTorch Pickle Error (Unpickling Issue)

🔹 Solution: Downgrade PyTorch to avoid strict safe-unpickling:

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --force-reinstall
```
---
## 📄 License

This project is based on YOLOv8 by Ultralytics.

Refer to the LICENSE for details.

--- 

## 🎯 Conclusion

YOLOv8 is trained on pothole datasets with YOLO format labels.

GPU acceleration speeds up training & detection.

Use TensorBoard for monitoring (tensorboard --logdir runs/train).

Adjust epochs, batch size, confidence, and IoU thresholds for better results.

---

## 🚀 Happy Training! 🏗️🔥


