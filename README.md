# DCSNet: Multiscale Feature Aggregation for Small Medical Object Segmentation
This repository contains the official PyTorch implementation of **DCSNet: Multiscale Feature Aggregation with Detection-guided Hierarchical Cropping** for small medical object segmentation.

## Introduction
Small medical object segmentation is challenging due to the extreme imbalance between the target and background. DCSNet introduces a Detection-guided Hierarchical Cropping mechanism combined with Multiscale Feature Aggregation to efficiently and accurately segment small targets (such as brain tumors, polyps, etc.) from medical images.

## Environment
- Python == 3.10.11
- PyTorch == 2.10.0
- torchvision
- numpy
- opencv-python


## Dataset Preparation
To train and test the model, you need to organize your medical image dataset into a specific folder structure. We recommend organizing your data into images and masks (ground truth) folders for training, validation, and testing.Due to the distinct characteristics of different medical datasets (e.g., varying image resolutions, target sizes, and background complexity), we provide tailored training and network scripts for each dataset.

### Repository Files
* **Brain Tumor:** `train_brain.py`, `net_brain.py` (Default resolution: 512x512)
* **Polyp:** `train_polyp.py`, `net_polyp.py` (Default resolution: 256x256)
* **Kidney Stone:** `train_kidney.py`, `net_kidney.py` (Default resolution: 256x256)

### Directory Structure
Please arrange your dataset in the following format:
 ```
Dataset/
├── train/
│ ├── images/ # Original medical images 
│ └── masks/ # Corresponding segmentation masks
├── val/
│ ├── images/
│ └── masks/
└── test/
├── images/
└── masks/
 ```

## Running the Model
Since the models are optimized for their respective datasets, please use the corresponding scripts to train and evaluate.

### Training
To train the model from scratch, run the specific training script for your target dataset. Ensure you modify the `data_dir` path inside the script before execution.
```bash
# For Brain Tumor Segmentation
python train_brain.py

# For Polyp Segmentation
python train_polyp.py

# For Kidney Stone Segmentation
python train_kidney.py
```

### Testing & Evaluation
Testing is integrated into the training scripts and runs automatically after the model finishes training by loading the best checkpoint.
For instance, in train_brain.py, the evaluate_test_with_metrics function will calculate Dice, IoU, Precision, HD95, and MAE.
```bash
# For Brain Tumor Segmentation
python test_brain.py
```

## Checkpoint
We do not upload model checkpoint files in this repo.

