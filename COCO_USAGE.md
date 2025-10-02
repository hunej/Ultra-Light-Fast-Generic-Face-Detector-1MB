# COCO Object Detection with 640x640 Input

This document explains how to use this model for COCO object detection with 640x640 square input size.

## Changes from Original Face Detection Model

1. **Input Size**: Changed from 4:3 aspect ratio (e.g., 320x240, 640x480) to square 640x640 input
2. **Task**: Changed from face detection to general COCO object detection
3. **Classes**: Changed from 2 classes (BACKGROUND, face) to 81 classes (BACKGROUND + 80 COCO classes)

## COCO Classes

The model now supports 80 COCO object categories including:
- People: person
- Vehicles: bicycle, car, motorcycle, airplane, bus, train, truck, boat
- Animals: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- Household items: chair, couch, bed, dining table, toilet, tv, laptop, etc.
- And many more...

See `models/coco-model-labels.txt` for the complete list.

## Training

To train the model on COCO dataset:

```bash
python train.py \
    --dataset_type coco \
    --datasets /path/to/coco/train2017 \
    --validation_dataset /path/to/coco/val2017 \
    --input_size 640 \
    --net RFB \
    --batch_size 32 \
    --num_epochs 200
```

### COCO Dataset Structure

The COCO dataset should be organized as follows:
```
/path/to/coco/
├── train2017/
│   ├── 000000000001.jpg
│   ├── 000000000002.jpg
│   └── ...
├── val2017/
│   ├── 000000000001.jpg
│   └── ...
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

## Inference

To run detection on images:

```bash
python detect_imgs.py \
    --net_type RFB \
    --input_size 640 \
    --path ./test_images \
    --threshold 0.5 \
    --candidate_size 1500
```

The detection results will show:
- Bounding boxes around detected objects
- Class labels (e.g., "person", "car", "dog")
- Confidence scores

## Converting to ONNX

To convert the trained model to ONNX format:

```bash
python convert_to_onnx.py
```

This will:
- Load the model from `models/pretrained/version-RFB-640.pth`
- Use 640x640 input size
- Export to `models/onnx/version-RFB-640.onnx`

## Configuration Details

### Image Size Configuration (fd_config.py)

For 640x640 input:
- Image size: `[640, 640]` (square)
- Feature map sizes: `[[80, 40, 20, 10], [80, 40, 20, 10]]`
- Shrinkage: `[[8.0, 16.0, 32.0, 64.0], [8.0, 16.0, 32.0, 64.0]]`

### Number of Classes

The model architecture automatically adapts to the number of classes:
- Face detection: 2 classes (BACKGROUND + face)
- COCO detection: 81 classes (BACKGROUND + 80 COCO classes)

## Notes

1. **Model Files**: You'll need to retrain the model on COCO dataset, as the pretrained face detection weights are not compatible with 81 classes.

2. **Memory Requirements**: 640x640 input requires more GPU memory than the original 320x240. Adjust batch size accordingly.

3. **Performance**: The model maintains its ultra-light architecture (~1MB) but now detects general objects instead of just faces.

4. **Aspect Ratio**: Unlike the original face detector which used 4:3 aspect ratio, this version uses square 1:1 aspect ratio (640x640) which is more suitable for general object detection.

## Example Training Command

```bash
# Train on COCO dataset with RFB network
python train.py \
    --dataset_type coco \
    --datasets /data/coco/train2017 \
    --validation_dataset /data/coco/val2017 \
    --input_size 640 \
    --net RFB \
    --batch_size 16 \
    --num_epochs 200 \
    --lr 0.01 \
    --checkpoint_folder ./models/coco_640/
```

## Troubleshooting

1. **Out of Memory**: Reduce batch size or use a smaller input size (e.g., 320)
2. **COCO Dataset Not Found**: Ensure the annotation JSON files are in the `annotations/` subdirectory
3. **Label Mismatch**: Make sure to use `models/coco-model-labels.txt` instead of `models/voc-model-labels.txt`
