"""
PASCAL VOC 2007 dataset parser and PyTorch Dataset class.
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import random
from collections import defaultdict

# Target classes
TARGET_CLASSES = ['person', 'car', 'dog']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(TARGET_CLASSES)}


def parse_voc_xml(xml_path: Path) -> Dict:
    """
    Parse PASCAL VOC XML annotation file.
    
    Args:
        xml_path: Path to XML annotation file
    
    Returns:
        Dictionary with image info and annotations
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get image info
    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    # Parse objects
    objects = []
    for obj in root.findall('object'):
        # Skip difficult objects
        difficult = obj.find('difficult')
        if difficult is not None and int(difficult.text) == 1:
            continue
        
        class_name = obj.find('name').text
        # Only keep target classes
        if class_name not in TARGET_CLASSES:
            continue
        
        bbox = obj.find('bndbox')
        x_min = float(bbox.find('xmin').text)
        y_min = float(bbox.find('ymin').text)
        x_max = float(bbox.find('xmax').text)
        y_max = float(bbox.find('ymax').text)
        
        # Convert to normalized (x, y, w, h) format
        x = ((x_min + x_max) / 2) / width
        y = ((y_min + y_max) / 2) / height
        w = (x_max - x_min) / width
        h = (y_max - y_min) / height
        
        objects.append({
            'class': class_name,
            'class_id': CLASS_TO_IDX[class_name],
            'bbox': [x, y, w, h]  # Normalized
        })
    
    return {
        'filename': filename,
        'width': width,
        'height': height,
        'objects': objects
    }


def load_voc_dataset(voc_dir: Path) -> List[Dict]:
    """
    Load all VOC annotations and filter for target classes.
    
    Args:
        voc_dir: Path to VOC2007 directory
    
    Returns:
        List of image annotations
    """
    annotations_dir = voc_dir / 'Annotations'
    jpeg_dir = voc_dir / 'JPEGImages'
    
    if not annotations_dir.exists():
        raise ValueError(f"Annotations directory not found: {annotations_dir}")
    if not jpeg_dir.exists():
        raise ValueError(f"JPEGImages directory not found: {jpeg_dir}")
    
    dataset = []
    xml_files = list(annotations_dir.glob('*.xml'))
    
    print(f"Loading {len(xml_files)} annotation files...")
    
    for xml_path in xml_files:
        try:
            annotation = parse_voc_xml(xml_path)
            
            # Only keep images with at least one target class
            if len(annotation['objects']) > 0:
                # Verify image exists
                img_path = jpeg_dir / annotation['filename']
                if img_path.exists():
                    annotation['image_path'] = img_path
                    dataset.append(annotation)
        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")
            continue
    
    print(f"Loaded {len(dataset)} images with target classes")
    return dataset


def split_dataset(
    dataset: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset: List of image annotations
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train, val, test) datasets
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Shuffle dataset
    shuffled = dataset.copy()
    random.shuffle(shuffled)
    
    # Calculate split indices
    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_set = shuffled[:n_train]
    val_set = shuffled[n_train:n_train + n_val]
    test_set = shuffled[n_train + n_val:]
    
    print(f"Dataset split:")
    print(f"  Train: {len(train_set)} images")
    print(f"  Val: {len(val_set)} images")
    print(f"  Test: {len(test_set)} images")
    
    return train_set, val_set, test_set


class VOCDataset(Dataset):
    """
    PyTorch Dataset for PASCAL VOC object detection.
    """
    
    def __init__(
        self,
        annotations: List[Dict],
        transform=None,
        input_size: int = 224
    ):
        """
        Initialize VOC dataset.
        
        Args:
            annotations: List of image annotations
            transform: Optional transform function (from augmentation.py)
            input_size: Target input size for images
        """
        self.annotations = annotations
        self.transform = transform
        self.input_size = input_size
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
                - image: Tensor of shape (3, H, W)
                - boxes: List of boxes in format (x, y, w, h) normalized
                - labels: List of class indices
                - original_size: Tuple of (width, height)
        """
        ann = self.annotations[idx]
        
        # Load image
        img_path = ann['image_path']
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)
        
        # Get bounding boxes and labels
        boxes = [obj['bbox'] for obj in ann['objects']]
        labels = [obj['class_id'] for obj in ann['objects']]
        
        original_size = (ann['width'], ann['height'])
        
        # Apply augmentation if provided
        if self.transform is not None:
            image_tensor, boxes, labels = self.transform(
                image_np, boxes, labels, original_size
            )
        else:
            # Just convert to tensor and normalize
            from src.augmentation import get_val_augmentation
            val_aug = get_val_augmentation(self.input_size)
            augmented = val_aug(
                image=image_np,
                bboxes=[[(x-w/2)*original_size[0], (y-h/2)*original_size[1],
                        (x+w/2)*original_size[0], (y+h/2)*original_size[1]]
                       for x, y, w, h in boxes],
                labels=labels
            )
            image_tensor = augmented['image']
            # Convert back to normalized format
            new_h, new_w = image_tensor.shape[1], image_tensor.shape[2]
            boxes = [[(x1+x2)/2/new_w, (y1+y2)/2/new_h, (x2-x1)/new_w, (y2-y1)/new_h]
                    for x1, y1, x2, y2 in augmented['bboxes']]
            labels = augmented['labels']
        
        return {
            'image': image_tensor,
            'boxes': torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.zeros((0, 4)),
            'labels': torch.tensor(labels, dtype=torch.long) if len(labels) > 0 else torch.zeros((0,), dtype=torch.long),
            'original_size': original_size
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for batching variable number of objects per image.
    
    Args:
        batch: List of samples from dataset
    
    Returns:
        Batched dictionary
    """
    images = torch.stack([item['image'] for item in batch])
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    original_sizes = [item['original_size'] for item in batch]
    
    return {
        'images': images,
        'boxes': boxes,  # List of tensors (variable length)
        'labels': labels,  # List of tensors (variable length)
        'original_sizes': original_sizes
    }


def create_dataloaders(
    voc_dir: Path,
    batch_size: int = 8,
    input_size: int = 224,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        voc_dir: Path to VOC2007 directory
        batch_size: Batch size
        input_size: Input image size
        num_workers: Number of data loading workers
        seed: Random seed
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load dataset
    dataset = load_voc_dataset(voc_dir)
    
    # Split dataset
    train_set, val_set, test_set = split_dataset(dataset, seed=seed)
    
    # Save splits for later use
    processed_dir = voc_dir.parent / 'processed'
    processed_dir.mkdir(exist_ok=True)
    
    import pickle
    with open(processed_dir / 'train_set.pkl', 'wb') as f:
        pickle.dump(train_set, f)
    with open(processed_dir / 'val_set.pkl', 'wb') as f:
        pickle.dump(val_set, f)
    with open(processed_dir / 'test_set.pkl', 'wb') as f:
        pickle.dump(test_set, f)
    
    # Create datasets
    from src.augmentation import get_train_augmentation, get_val_augmentation, apply_augmentation
    
    def train_transform(img, boxes, labels, orig_size):
        aug = get_train_augmentation(input_size)
        return apply_augmentation(img, boxes, labels, aug, orig_size)
    
    def val_transform(img, boxes, labels, orig_size):
        aug = get_val_augmentation(input_size)
        return apply_augmentation(img, boxes, labels, aug, orig_size)
    
    train_dataset = VOCDataset(train_set, transform=train_transform, input_size=input_size)
    val_dataset = VOCDataset(val_set, transform=val_transform, input_size=input_size)
    test_dataset = VOCDataset(test_set, transform=val_transform, input_size=input_size)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

