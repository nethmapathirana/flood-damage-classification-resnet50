# Flood Damage Classification using ResNet50

This project investigates the use of transfer learning for flood damage classification using drone imagery collected after Cyclone Gabrielle, New Zealand.

This repository contains a simplified implementation used for the ICTICM 2026 work-in-progress study.

## Overview

A pre-trained ResNet50 model was fine-tuned to classify drone images into:

- Damaged
- Non-Damaged

## Dataset

- Source: LINZ (Cyclone Gabrielle imagery)
- Total Images: 100
- Damaged: 60
- Non-Damaged: 40

## Data Split

- Training: 70%
- Testing: 30%

## Model

- ResNet50
- Transfer Learning
- Frozen feature extraction layers
- Modified final classification layer

## Training Configuration

- Optimizer: Adam
- Learning Rate: 0.001
- Batch Size: 8
- Epochs: 20

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Future Work

- Compare ResNet50 with U-Net and YOLOv8
- Expand dataset
- Integrate Sentinel-1 and Sentinel-2 imagery
- Object-level damage detection
