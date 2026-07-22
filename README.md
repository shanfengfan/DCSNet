# DCSNet
Multiscale Feature Aggregation for Small Medical Object Segmentation

## Introduction
This repository contains the official implementation of **DCSNet: Multiscale Feature Aggregation with Detection-guided Hierarchical Cropping** for small medical object segmentation.

## Environment
Python == 3.10.11
PyTorch == 2.10.0
torchvision
numpy
opencv-python


## Dataset
1. Prepare your medical dataset following the data format in `dataset.py`.
2. Modify the dataset path in the training script before running.


## Training
```bash
python train_brain.py

## Testing
python test_brain.py

## Checkpoint
We do not upload model checkpoint files in this repo.

