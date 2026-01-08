"""
Training script for object detection model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import os
from typing import Tuple

from src.model import create_model
from src.dataset import create_dataloaders
from src.utils import calculate_iou


class DetectionLoss(nn.Module):
    
    def __init__(self, bbox_weight: float = 1.0, class_weight: float = 1.0):
        super(DetectionLoss, self).__init__()
        self.bbox_weight = bbox_weight
        self.class_weight = class_weight
        self.bbox_loss_fn = nn.SmoothL1Loss(reduction='mean')
        self.class_loss_fn = nn.CrossEntropyLoss(reduction='mean')
    
    def forward(
        self,
        pred_bboxes: torch.Tensor,
        pred_classes: torch.Tensor,
        target_boxes: list,
        target_labels: list
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        batch_size = pred_bboxes.size(0)
        device = pred_bboxes.device
        
        total_bbox_loss = 0.0
        total_class_loss = 0.0
        num_valid_samples = 0
        
        for b in range(batch_size):
            target_boxes_b = target_boxes[b] 
            target_labels_b = target_labels[b] 
            
            if len(target_boxes_b) == 0:
                bg_logits = pred_classes[b, 0:1, :]  
                bg_target = torch.full((1,), 3, dtype=torch.long, device=device)  
                class_loss = self.class_loss_fn(bg_logits, bg_target)
                total_class_loss += class_loss
                num_valid_samples += 1
                continue
            
            num_objects = len(target_boxes_b)
            num_valid_samples += 1
            
            max_detections = pred_bboxes.size(1)
            
            if num_objects > max_detections:
                target_boxes_b = target_boxes_b[:max_detections]
                target_labels_b = target_labels_b[:max_detections]
                num_objects = max_detections
            
            pred_boxes_b = pred_bboxes[b, :num_objects]  
            pred_classes_b = pred_classes[b, :num_objects]  
            
            target_boxes_tensor = target_boxes_b.to(device)
            bbox_loss = self.bbox_loss_fn(pred_boxes_b, target_boxes_tensor)
            total_bbox_loss += bbox_loss
            
            target_labels_tensor = target_labels_b.to(device)
            class_loss = self.class_loss_fn(pred_classes_b, target_labels_tensor)
            total_class_loss += class_loss
            
            if num_objects < max_detections:
                remaining_preds = pred_classes[b, num_objects:]  
                bg_targets = torch.full(
                    (max_detections - num_objects,), 
                    3, 
                    dtype=torch.long, 
                    device=device
                )
                bg_class_loss = self.class_loss_fn(remaining_preds, bg_targets)
                total_class_loss += bg_class_loss
        
        if num_valid_samples > 0:
            avg_bbox_loss = total_bbox_loss / num_valid_samples
            avg_class_loss = total_class_loss / num_valid_samples
        else:
            avg_bbox_loss = torch.tensor(0.0, device=device)
            avg_class_loss = torch.tensor(0.0, device=device)
        
        total_loss = self.bbox_weight * avg_bbox_loss + self.class_weight * avg_class_loss
        
        return total_loss, avg_bbox_loss, avg_class_loss


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> dict:
    model.train()
    running_loss = 0.0
    running_bbox_loss = 0.0
    running_class_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, batch in enumerate(pbar):
        images = batch['images'].to(device)
        boxes = batch['boxes']
        labels = batch['labels']
        
        optimizer.zero_grad()
        pred_bboxes, pred_classes = model(images)
        
        loss, bbox_loss, class_loss = criterion(
            pred_bboxes, pred_classes, boxes, labels
        )
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_bbox_loss += bbox_loss.item()
        running_class_loss += class_loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'bbox': f'{bbox_loss.item():.4f}',
            'cls': f'{class_loss.item():.4f}'
        })
    
    return {
        'loss': running_loss / len(dataloader),
        'bbox_loss': running_bbox_loss / len(dataloader),
        'class_loss': running_class_loss / len(dataloader)
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> dict:
    model.eval()
    running_loss = 0.0
    running_bbox_loss = 0.0
    running_class_loss = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
        for batch in pbar:
            images = batch['images'].to(device)
            boxes = batch['boxes']
            labels = batch['labels']
            
            pred_bboxes, pred_classes = model(images)
            
            loss, bbox_loss, class_loss = criterion(
                pred_bboxes, pred_classes, boxes, labels
            )
            
            running_loss += loss.item()
            running_bbox_loss += bbox_loss.item()
            running_class_loss += class_loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'bbox': f'{bbox_loss.item():.4f}',
                'cls': f'{class_loss.item():.4f}'
            })
    
    return {
        'loss': running_loss / len(dataloader),
        'bbox_loss': running_bbox_loss / len(dataloader),
        'class_loss': running_class_loss / len(dataloader)
    }


def main():
    parser = argparse.ArgumentParser(description='Train object detection model')
    parser.add_argument('--data_dir', type=str, default='data/voc2007',
                        help='Path to VOC2007 dataset')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--input_size', type=int, default=224,
                        help='Input image size (default: 224)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers (default: 0)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    print('Loading dataset...')
    voc_dir = Path(args.data_dir)
    train_loader, val_loader, test_loader = create_dataloaders(
        voc_dir=voc_dir,
        batch_size=args.batch_size,
        input_size=args.input_size,
        num_workers=args.num_workers
    )
    
    print('Creating model...')
    model = create_model(num_classes=3, input_size=args.input_size)
    model = model.to(device)
    
    criterion = DetectionLoss(bbox_weight=1.0, class_weight=1.0)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
    
    print('Starting training...')
    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        val_metrics = validate(
            model, val_loader, criterion, device, epoch
        )
        
        print(f'\nEpoch {epoch} Summary:')
        print(f'  Train Loss: {train_metrics["loss"]:.4f} '
              f'(Bbox: {train_metrics["bbox_loss"]:.4f}, '
              f'Class: {train_metrics["class_loss"]:.4f})')
        print(f'  Val Loss: {val_metrics["loss"]:.4f} '
              f'(Bbox: {val_metrics["bbox_loss"]:.4f}, '
              f'Class: {val_metrics["class_loss"]:.4f})')
        
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
            print(f'  New best validation loss: {best_val_loss:.4f}')
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'best_val_loss': best_val_loss
        }
        
        torch.save(checkpoint, checkpoint_dir / 'latest_checkpoint.pth')
        
        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
            print(f'  Saved best model to: {checkpoint_dir / "best_model.pth"}')
        
        print('-' * 60)
    
    print('Training completed!')
    print(f'Best validation loss: {best_val_loss:.4f}')
    print(f'Best model saved to: {checkpoint_dir / "best_model.pth"}')


if __name__ == '__main__':
    main()

