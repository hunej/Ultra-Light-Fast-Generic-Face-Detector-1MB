import logging
import os
import pathlib

import cv2
import numpy as np
from pycocotools.coco import COCO


class COCODataset:
    """Dataset for COCO data.
    
    This dataset class loads COCO format annotations and images for object detection.
    It expects the standard COCO directory structure with images and annotations.
    """

    def __init__(self, root, ann_file=None, transform=None, target_transform=None, is_test=False, keep_difficult=False):
        """Initialize COCO Dataset.
        
        Args:
            root: Root directory containing the COCO dataset images
            ann_file: Path to the COCO annotation JSON file. If None, will look in root/annotations/
            transform: Image transformation function
            target_transform: Target (boxes, labels) transformation function
            is_test: Whether this is a test dataset
            keep_difficult: Whether to keep difficult samples (not used in COCO)
        """
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.keep_difficult = keep_difficult
        
        # Determine annotation file
        if ann_file is None:
            if is_test:
                ann_file = self.root / "annotations" / "instances_val2017.json"
            else:
                ann_file = self.root / "annotations" / "instances_train2017.json"
        else:
            ann_file = pathlib.Path(ann_file)
            
        if not ann_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")
            
        # Load COCO annotations
        self.coco = COCO(str(ann_file))
        
        # Get all image IDs
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        # COCO has 80 classes, but they are not contiguous (1-90 with gaps)
        # We need to map COCO category IDs to contiguous indices
        coco_categories = self.coco.loadCats(self.coco.getCatIds())
        coco_categories = sorted(coco_categories, key=lambda x: x['id'])
        
        # Create class names tuple (prepend BACKGROUND)
        self.class_names = ('BACKGROUND',) + tuple(cat['name'] for cat in coco_categories)
        
        # Create mapping from COCO category ID to our class index
        self.coco_id_to_class_idx = {cat['id']: i + 1 for i, cat in enumerate(coco_categories)}
        
        logging.info(f"COCO Dataset initialized with {len(self.ids)} images and {len(self.class_names)} classes")

    def __getitem__(self, index):
        """Get item at index.
        
        Returns:
            image: Image as numpy array
            boxes: Bounding boxes as numpy array [N, 4] in format [xmin, ymin, xmax, ymax]
            labels: Class labels as numpy array [N]
        """
        image_id = self.ids[index]
        boxes, labels = self._get_annotation(image_id)
        image = self._read_image(image_id)
        
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
            
        return image, boxes, labels

    def get_image(self, index):
        """Get image at index without annotations."""
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        """Get annotation for image at index."""
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    def _get_annotation(self, image_id):
        """Get annotations for a specific image ID.
        
        Args:
            image_id: COCO image ID
            
        Returns:
            boxes: numpy array of shape [N, 4] with bounding boxes in [xmin, ymin, xmax, ymax] format
            labels: numpy array of shape [N] with class indices
        """
        ann_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)
        
        boxes = []
        labels = []
        
        for ann in anns:
            # COCO bbox format is [x, y, width, height]
            x, y, w, h = ann['bbox']
            # Convert to [xmin, ymin, xmax, ymax]
            boxes.append([x, y, x + w, y + h])
            # Map COCO category ID to our class index
            labels.append(self.coco_id_to_class_idx[ann['category_id']])
        
        if len(boxes) == 0:
            # Image has no annotations, return empty arrays
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
        else:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
            
        return boxes, labels

    def _read_image(self, image_id):
        """Read image from disk.
        
        Args:
            image_id: COCO image ID
            
        Returns:
            image: numpy array in RGB format
        """
        img_info = self.coco.loadImgs(image_id)[0]
        image_file = self.root / img_info['file_name']
        
        if not image_file.exists():
            # Try alternative path structures
            # Sometimes images are in train2017/ or val2017/ subdirectories
            for subdir in ['train2017', 'val2017', 'test2017']:
                alt_path = self.root / subdir / img_info['file_name']
                if alt_path.exists():
                    image_file = alt_path
                    break
        
        image = cv2.imread(str(image_file))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_file}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
