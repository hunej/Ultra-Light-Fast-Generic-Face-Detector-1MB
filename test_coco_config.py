#!/usr/bin/env python3
"""
Quick test script to verify 640x640 COCO configuration
Run this after setup to ensure everything is working correctly.
"""

import torch
from vision.ssd.config.fd_config import define_img_size

# Configure for 640x640
define_img_size(640)
from vision.ssd.config import fd_config
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd

# Load COCO labels
with open('models/coco-model-labels.txt', 'r') as f:
    coco_classes = [line.strip() for line in f.readlines()]

print("=" * 70)
print("COCO Object Detection Configuration - 640x640")
print("=" * 70)

# Test configuration
print(f"\n✓ Input size: {fd_config.image_size[0]}x{fd_config.image_size[1]}")
print(f"✓ Number of classes: {len(coco_classes)}")
print(f"✓ Number of anchor priors: {len(fd_config.priors)}")

# Test network
print("\n✓ Creating network with 81 classes (BACKGROUND + 80 COCO)...")
net = create_Mb_Tiny_RFB_fd(len(coco_classes), is_test=True, device='cpu')

# Test forward pass
print("✓ Testing forward pass with 640x640 input...")
dummy_input = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    scores, boxes = net(dummy_input)

print(f"  - Input shape: {list(dummy_input.shape)}")
print(f"  - Scores shape: {list(scores.shape)}")
print(f"  - Boxes shape: {list(boxes.shape)}")

print("\n✓ Sample COCO classes:")
for i in range(1, 11):
    print(f"  {i}. {coco_classes[i]}")

print("\n" + "=" * 70)
print("Configuration verified successfully!")
print("=" * 70)
print("\nNext steps:")
print("1. Prepare COCO dataset in standard format")
print("2. Train with: python train.py --dataset_type coco --input_size 640 ...")
print("3. See COCO_USAGE.md for detailed instructions")
