"""
Training script for object detection model.
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
    """
    Combined loss for object detection.
    - Smooth L1 Loss for bounding boxes
    - Cross Entropy Loss for classification
    """
    
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
        """
        Calculate detection loss.
        
        Args:
            pred_bboxes: Predicted boxes (B, max_detections, 4)
            pred_classes: Predicted class logits (B, max_detections, num_classes+1)
            target_boxes: List of target boxes per image (variable length)
            target_labels: List of target labels per image (variable length)
        
        Returns:
            Total loss, bbox loss, class loss
        """
        batch_size = pred_bboxes.size(0)
        device = pred_bboxes.device
        
        total_bbox_loss = 0.0
        total_class_loss = 0.0
        num_valid_samples = 0
        
        for b in range(batch_size):
            target_boxes_b = target_boxes[b]  # (N, 4)
            target_labels_b = target_labels[b]  # (N,)
            
            if len(target_boxes_b) == 0:
                # No objects in this image - penalize background predictions
                # Use first detection as background
                bg_logits = pred_classes[b, 0:1, :]  # (1, num_classes+1)
                bg_target = torch.full((1,), 3, dtype=torch.long, device=device)  # Background class is last (index 3)
                class_loss = self.class_loss_fn(bg_logits, bg_target)
                total_class_loss += class_loss
                num_valid_samples += 1
                continue
            
            num_objects = len(target_boxes_b)
            num_valid_samples += 1
            
            # Match predictions to targets using IoU
            # For simplicity, use first N predictions for N targets
            # In practice, you might want to use Hungarian matching or similar
            pred_boxes_b = pred_bboxes[b, :num_objects]  # (N, 4)
            pred_classes_b = pred_classes[b, :num_objects]  # (N, num_classes+1)
            
            # Bounding box loss
            target_boxes_tensor = target_boxes_b.to(device)
            bbox_loss = self.bbox_loss_fn(pred_boxes_b, target_boxes_tensor)
            total_bbox_loss += bbox_loss
            
            # Classification loss
            # Add background class (index = num_classes) for remaining predictions
            target_labels_tensor = target_labels_b.to(device)
            class_loss = self.class_loss_fn(pred_classes_b, target_labels_tensor)
            total_class_loss += class_loss
        
        # Average losses
        if num_valid_samples > 0:
            avg_bbox_loss = total_bbox_loss / num_valid_samples
            avg_class_loss = total_class_loss / num_valid_samples
        else:
            avg_bbox_loss = torch.tensor(0.0, device=device)
            avg_class_loss = torch.tensor(0.0, device=device)
        
        # Combined loss
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
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    running_bbox_loss = 0.0
    running_class_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, batch in enumerate(pbar):
        images = batch['images'].to(device)
        boxes = batch['boxes']
        labels = batch['labels']
        
        # Forward pass
        optimizer.zero_grad()
        pred_bboxes, pred_classes = model(images)
        
        # Calculate loss
        loss, bbox_loss, class_loss = criterion(
            pred_bboxes, pred_classes, boxes, labels
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update statistics
        running_loss += loss.item()
        running_bbox_loss += bbox_loss.item()
        running_class_loss += class_loss.item()
        
        # Update progress bar
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
    """Validate model."""
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
            
            # Forward pass
            pred_bboxes, pred_classes = model(images)
            
            # Calculate loss
            loss, bbox_loss, class_loss = criterion(
                pred_bboxes, pred_classes, boxes, labels
            )
            
            # Update statistics
            running_loss += loss.item()
            running_bbox_loss += bbox_loss.item()
            running_class_loss += class_loss.item()
            
            # Update progress bar
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
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Create dataloaders
    print('Loading dataset...')
    voc_dir = Path(args.data_dir)
    train_loader, val_loader, test_loader = create_dataloaders(
        voc_dir=voc_dir,
        batch_size=args.batch_size,
        input_size=args.input_size,
        num_workers=args.num_workers
    )
    
    # Create model
    print('Creating model...')
    model = create_model(num_classes=3, input_size=args.input_size)
    model = model.to(device)
    
    # Create loss function
    criterion = DetectionLoss(bbox_weight=1.0, class_weight=1.0)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
    
    # Training loop
    print('Starting training...')
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Print epoch summary
        print(f'\nEpoch {epoch} Summary:')
        print(f'  Train Loss: {train_metrics["loss"]:.4f} '
              f'(Bbox: {train_metrics["bbox_loss"]:.4f}, '
              f'Class: {train_metrics["class_loss"]:.4f})')
        print(f'  Val Loss: {val_metrics["loss"]:.4f} '
              f'(Bbox: {val_metrics["bbox_loss"]:.4f}, '
              f'Class: {val_metrics["class_loss"]:.4f})')
        
        # Save checkpoint
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
        
        # Save latest checkpoint
        torch.save(checkpoint, checkpoint_dir / 'latest_checkpoint.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
            print(f'  Saved best model to: {checkpoint_dir / "best_model.pth"}')
        
        print('-' * 60)
    
    print('Training completed!')
    print(f'Best validation loss: {best_val_loss:.4f}')
    print(f'Best model saved to: {checkpoint_dir / "best_model.pth"}')


if __name__ == '__main__':
    main()

