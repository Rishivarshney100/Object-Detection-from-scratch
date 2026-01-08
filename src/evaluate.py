"""
Evaluation script for object detection model
Computes mAP@0.5, FPS, and model size
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import time
import os
from tqdm import tqdm
import numpy as np

from torchmetrics.detection import MeanAveragePrecision
from src.model import create_model
from src.dataset import create_dataloaders
from src.utils import nms, denormalize_boxes, calculate_iou


def evaluate_map(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5
) -> float:
    
    model.eval()
    
    map_metric = MeanAveragePrecision(
        box_format='cxcywh',  
        iou_type='bbox',
        iou_thresholds=[iou_threshold]
    )
    
    class_names = ['person', 'car', 'dog']
    
    print('Computing mAP@0.5...')
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            images = batch['images'].to(device)
            target_boxes = batch['boxes']
            target_labels = batch['labels']
            original_sizes = batch['original_sizes']
            
            pred_bboxes, pred_classes = model(images)
            
            batch_size = images.size(0)
            for b in range(batch_size):
                pred_boxes_b = pred_bboxes[b].cpu()  
                pred_scores_b = torch.softmax(pred_classes[b], dim=1).cpu()  
                
                max_scores, class_indices = torch.max(pred_scores_b[:, :3], dim=1)
                
                valid_mask = max_scores >= score_threshold
                if valid_mask.sum() == 0:
                    pred_dict = {
                        'boxes': torch.zeros((0, 4), dtype=torch.float),
                        'scores': torch.zeros((0,), dtype=torch.float),
                        'labels': torch.zeros((0,), dtype=torch.long)
                    }
                else:
                    pred_boxes_filtered = pred_boxes_b[valid_mask]
                    pred_scores_filtered = max_scores[valid_mask]
                    pred_labels_filtered = class_indices[valid_mask]
                    
                    if len(pred_boxes_filtered) > 0:
                        keep_indices = nms(pred_boxes_filtered, pred_scores_filtered, iou_threshold=0.5)
                        pred_boxes_filtered = pred_boxes_filtered[keep_indices]
                        pred_scores_filtered = pred_scores_filtered[keep_indices]
                        pred_labels_filtered = pred_labels_filtered[keep_indices]
                    
                    img_width, img_height = original_sizes[b]
                    pred_boxes_pixel = denormalize_boxes(
                        pred_boxes_filtered, img_width, img_height
                    )
                    
                    pred_dict = {
                        'boxes': pred_boxes_pixel,
                        'scores': pred_scores_filtered,
                        'labels': pred_labels_filtered
                    }
                
                target_boxes_b = target_boxes[b].cpu()
                target_labels_b = target_labels[b].cpu()
                
                if len(target_boxes_b) > 0:
                    img_width, img_height = original_sizes[b]
                    target_boxes_pixel = denormalize_boxes(
                        target_boxes_b, img_width, img_height
                    )
                    
                    target_dict = {
                        'boxes': target_boxes_pixel,
                        'labels': target_labels_b
                    }
                else:
                    target_dict = {
                        'boxes': torch.zeros((0, 4), dtype=torch.float),
                        'labels': torch.zeros((0,), dtype=torch.long)
                    }
                
                map_metric.update([pred_dict], [target_dict])
    
    map_result = map_metric.compute()
    map_value = map_result['map'].item()
    
    return map_value


def evaluate_fps(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_warmup: int = 10,
    num_runs: int = 100
) -> float:
    
    model.eval()
    
    print(f'Warming up ({num_warmup} runs)...')
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_warmup:
                break
            images = batch['images'].to(device)
            _ = model(images)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    print(f'Measuring FPS ({num_runs} runs)...')
    total_time = 0.0
    total_frames = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_runs:
                break
            
            images = batch['images'].to(device)
            batch_size = images.size(0)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            _ = model(images)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            elapsed = end_time - start_time
            total_time += elapsed
            total_frames += batch_size
    
    fps = total_frames / total_time if total_time > 0 else 0.0
    
    return fps


def get_model_size(model_path: Path) -> float:
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)
    
    return size_mb


def main():
    parser = argparse.ArgumentParser(description='Evaluate object detection model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/voc2007',
                        help='Path to VOC2007 dataset')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--input_size', type=int, default=224,
                        help='Input image size (default: 224)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers (default: 0)')
    parser.add_argument('--score_threshold', type=float, default=0.5,
                        help='Score threshold for detections (default: 0.5)')
    parser.add_argument('--skip_map', action='store_true',
                        help='Skip mAP calculation')
    parser.add_argument('--skip_fps', action='store_true',
                        help='Skip FPS calculation')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    print(f'Loading model from: {args.model_path}')
    model = create_model(num_classes=3, input_size=args.input_size)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print('Loading test dataset...')
    voc_dir = Path(args.data_dir)
    
    import pickle
    processed_dir = voc_dir.parent / 'processed'
    if (processed_dir / 'test_set.pkl').exists():
        with open(processed_dir / 'test_set.pkl', 'rb') as f:
            test_set = pickle.load(f)
        
        from src.dataset import VOCDataset, collate_fn
        from src.augmentation import get_val_augmentation, apply_augmentation
        
        def val_transform(img, boxes, labels, orig_size):
            aug = get_val_augmentation(args.input_size)
            return apply_augmentation(img, boxes, labels, aug, orig_size)
        
        test_dataset = VOCDataset(test_set, transform=val_transform, input_size=args.input_size)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    else:
        _, _, test_loader = create_dataloaders(
            voc_dir=voc_dir,
            batch_size=args.batch_size,
            input_size=args.input_size,
            num_workers=args.num_workers
        )
    
    print('=' * 60)
    print('EVALUATION RESULTS')
    print('=' * 60)
    
    if not args.skip_map:
        print('\n1. Computing mAP@0.5...')
        map_value = evaluate_map(
            model, test_loader, device, iou_threshold=0.5, score_threshold=args.score_threshold
        )
        print(f'   mAP@0.5: {map_value:.4f}')
    else:
        map_value = None
        print('\n1. mAP@0.5: Skipped')
    
    if not args.skip_fps:
        print('\n2. Computing FPS...')
        fps = evaluate_fps(model, test_loader, device)
        print(f'   FPS: {fps:.2f} frames/second')
    else:
        fps = None
        print('\n2. FPS: Skipped')
    
    print('\n3. Computing model size...')
    model_path = Path(args.model_path)
    model_size = get_model_size(model_path)
    print(f'   Model Size: {model_size:.2f} MB')
    
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    if map_value is not None:
        print(f'mAP@0.5: {map_value:.4f}')
    if fps is not None:
        print(f'FPS: {fps:.2f}')
    print(f'Model Size: {model_size:.2f} MB')
    print('=' * 60)


if __name__ == '__main__':
    main()

