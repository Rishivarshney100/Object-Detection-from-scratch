"""
PASCAL VOC 2007 dataset parser and PyTorch Dataset class
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

TARGET_CLASSES = ['person', 'car', 'dog']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(TARGET_CLASSES)}


def parse_voc_xml(xml_path: Path) -> Dict:
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    objects = []
    for obj in root.findall('object'):
        difficult = obj.find('difficult')
        if difficult is not None and int(difficult.text) == 1:
            continue
        
        class_name = obj.find('name').text
        if class_name not in TARGET_CLASSES:
            continue
        
        bbox = obj.find('bndbox')
        x_min = float(bbox.find('xmin').text)
        y_min = float(bbox.find('ymin').text)
        x_max = float(bbox.find('xmax').text)
        y_max = float(bbox.find('ymax').text)
        
        x = ((x_min + x_max) / 2) / width
        y = ((y_min + y_max) / 2) / height
        w = (x_max - x_min) / width
        h = (y_max - y_min) / height
        
        objects.append({
            'class': class_name,
            'class_id': CLASS_TO_IDX[class_name],
            'bbox': [x, y, w, h]  
        })
    
    return {
        'filename': filename,
        'width': width,
        'height': height,
        'objects': objects
    }


def load_voc_dataset(voc_dir: Path) -> List[Dict]:
    
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
            
            if len(annotation['objects']) > 0:
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
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    random.seed(seed)
    np.random.seed(seed)
    
    shuffled = dataset.copy()
    random.shuffle(shuffled)
    
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
    
    def __init__(
        self,
        annotations: List[Dict],
        transform=None,
        input_size: int = 224
    ):
        
        self.annotations = annotations
        self.transform = transform
        self.input_size = input_size
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict:
        
        ann = self.annotations[idx]
        
        img_path = ann['image_path']
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)
        
        boxes = [obj['bbox'] for obj in ann['objects']]
        labels = [obj['class_id'] for obj in ann['objects']]
        
        original_size = (ann['width'], ann['height'])
        
        if self.transform is not None:
            image_tensor, boxes, labels = self.transform(
                image_np, boxes, labels, original_size
            )
        else:
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
    
    images = torch.stack([item['image'] for item in batch])
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    original_sizes = [item['original_size'] for item in batch]
    
    return {
        'images': images,
        'boxes': boxes,  
        'labels': labels,  
        'original_sizes': original_sizes
    }


def create_dataloaders(
    voc_dir: Path,
    batch_size: int = 8,
    input_size: int = 224,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    dataset = load_voc_dataset(voc_dir)
    
    train_set, val_set, test_set = split_dataset(dataset, seed=seed)
    
    processed_dir = voc_dir.parent / 'processed'
    processed_dir.mkdir(exist_ok=True)
    
    import pickle
    with open(processed_dir / 'train_set.pkl', 'wb') as f:
        pickle.dump(train_set, f)
    with open(processed_dir / 'val_set.pkl', 'wb') as f:
        pickle.dump(val_set, f)
    with open(processed_dir / 'test_set.pkl', 'wb') as f:
        pickle.dump(test_set, f)
    
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

