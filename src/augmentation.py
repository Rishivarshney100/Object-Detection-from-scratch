"""
Data augmentation for object detection with bounding box handling
"""
import torch
import numpy as np
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, List, Dict


def get_train_augmentation(input_size: int = 224) -> A.Compose:
    
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomScale(scale_limit=0.2, p=0.5),
            A.Rotate(limit=10, p=0.5, border_mode=0),
            A.Resize(height=input_size, width=input_size, p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.0,
                p=0.5
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_visibility=0.3
        )
    )


def get_val_augmentation(input_size: int = 224) -> A.Compose:
    
    return A.Compose(
        [
            # Resize to exact size (ensures all images are same size)
            A.Resize(height=input_size, width=input_size, p=1.0),
            # Normalize
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels']
        )
    )


def convert_bbox_to_pascal_voc(bboxes: List[List[float]], img_width: int, img_height: int) -> List[List[float]]:
    
    pascal_boxes = []
    for bbox in bboxes:
        x, y, w, h = bbox
        x_min = (x - w/2) * img_width
        y_min = (y - h/2) * img_height
        x_max = (x + w/2) * img_width
        y_max = (y + h/2) * img_height
        pascal_boxes.append([x_min, y_min, x_max, y_max])
    return pascal_boxes


def convert_bbox_from_pascal_voc(bboxes: List[List[float]], img_width: int, img_height: int) -> List[List[float]]:
    
    normalized_boxes = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        x = ((x_min + x_max) / 2) / img_width
        y = ((y_min + y_max) / 2) / img_height
        w = (x_max - x_min) / img_width
        h = (y_max - y_min) / img_height
        normalized_boxes.append([x, y, w, h])
    return normalized_boxes


def apply_augmentation(
    image: np.ndarray,
    bboxes: List[List[float]],
    labels: List[int],
    augmentation: A.Compose,
    original_size: Tuple[int, int]
) -> Tuple[torch.Tensor, List[List[float]], List[int]]:
    
    img_width, img_height = original_size
    
    pascal_boxes = convert_bbox_to_pascal_voc(bboxes, img_width, img_height)
    
    augmented = augmentation(
        image=image,
        bboxes=pascal_boxes,
        labels=labels
    )
    
    aug_image = augmented['image']
    
    aug_boxes_pascal = augmented['bboxes']
    aug_labels = augmented['labels']
    
    new_height, new_width = aug_image.shape[1], aug_image.shape[2]
    
    aug_boxes = convert_bbox_from_pascal_voc(aug_boxes_pascal, new_width, new_height)
    
    return aug_image, aug_boxes, aug_labels

