# Cross-Domain Generalization Limits of Vision Foundation Models in Facial Deepfake Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg)](https://pytorch.org/)

This repository contains the official evaluation pipeline and trained downstream linear probing weights (`_LP`) for our paper exploring the cross-domain generalization boundaries of major Vision Foundation Models (VFMs) in digital forensics. 

We provide complete inference support for our top three localized probing architectures:
1. **Supervised ViT** (`vit_base_patch16_rope_mixed_ape_224.naver_in1k`)
2. **DINOv3** (`vit_large_patch16_dinov3.lvd1689m`)
3. **NVIDIA RADIOv4** (`c_radiov4_h`)

---

## 📌 Architectural & Labeling Configurations

To maintain exact reproducibility with the benchmarks established in our paper, please note the following operational constraints:
* **Label Mapping:** Binary classification targets are configured as **`REAL = 1`** and **`FAKE = 0`** (Sigmoid output closer to `1.0` denotes higher authentic confidence).
* **Resolution Scaling:** Input facial regions of interest (ROIs) are processed at a normalized input scale of **224x224 pixels** with standard ImageNet normalization coordinates.
* **Feature Aggregation:** Features are gathered directly from the patch output space via `forward_features()` (or the summary layer outputs for RADIOv4) and spatial sequence dimensions are globally mean-pooled prior to linear classification.

---

## ⚙️ Installation & Workspace Setup

### 1. Environment Prerequisites
Install the required feature engineering libraries, network backbones, and dashboard presentation utilities:
```
pip install torch torchvision timm pillow tabulate huggingface_hub
```

### 2. Retrieve Offline VFM Parameters (Mandatory First Step)
Before running any inference routines, you must download the native weight tensors for the NVIDIA C-RADIOv4-H core framework. Run the checkpoint download utility script first to pull down the necessary configuration properties and safe-tensors completely offline into your local directory structure:

```
python download_backbone.py
```
### 3. Execution & Processing Pipeline
Step 1: Extract Face Region of Interest (Required Preprocessing)
Deepfake manipulation artifacts are intensely localized around facial blending edges, internal micro-textures, and key geometric boundaries. To filter out background noise that causes distributional shift in foundation models, you must run the face detection script first to isolate and crop the target facial bounding box from raw source images before running classification:

```
python detect-faces.py --root_dir sample_dataset
```


Step 2: Running Inference Commands
Scenario A: Standalone Input Image Inference
To evaluate a single preprocessed facial crop (.jpg, .png, etc.) and inspect the descriptive prediction scores across all three linear probes simultaneously:

```
python inference.py --input ibrahim.jpg
```
Scenario B: Dataset Folder Evaluation
To calculate standard verification metrics (Accuracy, Precision, Recall, and F1-Score) across an entire test split, organize your dataset folder branches using the following subfolder structure:

```
target_dataset/
├── real/   <- Target Label: 1
│   ├── sample_001.png
│   └── sample_002.jpg
└── fake/   <- Target Label: 0
    ├── sample_003.png
    └── sample_004.jpg
```

Execute the batch evaluation pipeline by passing the directory path directly:

```
python inference.py --input sample_dataset
```
