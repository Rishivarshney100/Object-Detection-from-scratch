"""
Data augmentation for object detection with bounding box handling.
"""
import torch
import numpy as np
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, List, Dict


def get_train_augmentation(input_size: int = 224) -> A.Compose:
    """
    Get training augmentation pipeline.
    
    Augmentations:
    - Horizontal flip (with bbox update)
    - Random scaling (0.8-1.2x) with bbox scaling
    - Brightness adjustment (±20%) - no bbox change
    - Small rotation (±10°) with bbox rotation
    
    Args:
        input_size: Target image size
    
    Returns:
        Albumentations compose object
    """
    return A.Compose(
        [
            # Resize to input size
            A.LongestMaxSize(max_size=input_size, p=1.0),
            A.PadIfNeeded(
                min_height=input_size,
                min_width=input_size,
                border_mode=0,
                value=0,
                p=1.0
            ),
            # Horizontal flip
            A.HorizontalFlip(p=0.5),
            # Random scaling (0.8-1.2x)
            A.RandomScale(scale_limit=0.2, p=0.5),
            # Brightness adjustment
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.0,
                p=0.5
            ),
            # Small rotation (±10°)
            A.Rotate(limit=10, p=0.5, border_mode=0, value=0),
            # Normalize
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',  # (x_min, y_min, x_max, y_max)
            label_fields=['labels'],
            min_visibility=0.3  # Remove boxes with < 30% visibility after augmentation
        )
    )


def get_val_augmentation(input_size: int = 224) -> A.Compose:
    """
    Get validation/test augmentation pipeline (only resize and normalize).
    
    Args:
        input_size: Target image size
    
    Returns:
        Albumentations compose object
    """
    return A.Compose(
        [
            # Resize to input size
            A.LongestMaxSize(max_size=input_size, p=1.0),
            A.PadIfNeeded(
                min_height=input_size,
                min_width=input_size,
                border_mode=0,
                value=0,
                p=1.0
            ),
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
    """
    Convert normalized (x, y, w, h) boxes to Pascal VOC format (x_min, y_min, x_max, y_max).
    
    Args:
        bboxes: List of boxes in format (x, y, w, h) normalized [0, 1]
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        List of boxes in Pascal VOC format (x_min, y_min, x_max, y_max) in pixels
    """
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
    """
    Convert Pascal VOC format (x_min, y_min, x_max, y_max) to normalized (x, y, w, h).
    
    Args:
        bboxes: List of boxes in Pascal VOC format (x_min, y_min, x_max, y_max) in pixels
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        List of boxes in format (x, y, w, h) normalized [0, 1]
    """
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
    """
    Apply augmentation to image and bounding boxes.
    
    Args:
        image: Image as numpy array (H, W, 3) in RGB format
        bboxes: List of boxes in format (x, y, w, h) normalized [0, 1]
        labels: List of class labels
        augmentation: Albumentations compose object
        original_size: Original image size (width, height)
    
    Returns:
        Augmented image tensor, augmented bboxes, and labels
    """
    img_width, img_height = original_size
    
    # Convert to Pascal VOC format for Albumentations
    pascal_boxes = convert_bbox_to_pascal_voc(bboxes, img_width, img_height)
    
    # Apply augmentation
    augmented = augmentation(
        image=image,
        bboxes=pascal_boxes,
        labels=labels
    )
    
    # Get augmented image
    aug_image = augmented['image']
    
    # Get augmented boxes and labels (filtered by visibility)
    aug_boxes_pascal = augmented['bboxes']
    aug_labels = augmented['labels']
    
    # Get new image size
    new_height, new_width = aug_image.shape[1], aug_image.shape[2]
    
    # Convert back to normalized (x, y, w, h) format
    aug_boxes = convert_bbox_from_pascal_voc(aug_boxes_pascal, new_width, new_height)
    
    return aug_image, aug_boxes, aug_labels

