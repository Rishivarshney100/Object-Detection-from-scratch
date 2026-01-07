"""
Utility functions for object detection.
"""
import torch
import numpy as np
import cv2
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def calculate_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Calculate Intersection over Union (IoU) between two sets of boxes.
    
    Args:
        box1: Tensor of shape (N, 4) with format (x, y, w, h) normalized
        box2: Tensor of shape (M, 4) with format (x, y, w, h) normalized
    
    Returns:
        IoU tensor of shape (N, M)
    """
    # Convert (x, y, w, h) to (x1, y1, x2, y2)
    box1_xyxy = torch.zeros_like(box1)
    box1_xyxy[:, 0] = box1[:, 0] - box1[:, 2] / 2  # x1
    box1_xyxy[:, 1] = box1[:, 1] - box1[:, 3] / 2  # y1
    box1_xyxy[:, 2] = box1[:, 0] + box1[:, 2] / 2  # x2
    box1_xyxy[:, 3] = box1[:, 1] + box1[:, 3] / 2  # y2
    
    box2_xyxy = torch.zeros_like(box2)
    box2_xyxy[:, 0] = box2[:, 0] - box2[:, 2] / 2  # x1
    box2_xyxy[:, 1] = box2[:, 1] - box2[:, 3] / 2  # y1
    box2_xyxy[:, 2] = box2[:, 0] + box2[:, 2] / 2  # x2
    box2_xyxy[:, 3] = box2[:, 1] + box2[:, 3] / 2  # y2
    
    # Calculate intersection
    inter_x1 = torch.max(box1_xyxy[:, 0:1], box2_xyxy[:, 0].unsqueeze(0))
    inter_y1 = torch.max(box1_xyxy[:, 1:2], box2_xyxy[:, 1].unsqueeze(0))
    inter_x2 = torch.min(box1_xyxy[:, 2:3], box2_xyxy[:, 2].unsqueeze(0))
    inter_y2 = torch.min(box1_xyxy[:, 3:4], box2_xyxy[:, 3].unsqueeze(0))
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Calculate union
    box1_area = box1[:, 2] * box1[:, 3]
    box2_area = box2[:, 2] * box2[:, 3]
    union_area = box1_area.unsqueeze(1) + box2_area.unsqueeze(0) - inter_area
    
    # Calculate IoU
    iou = inter_area / (union_area + 1e-6)
    return iou


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> torch.Tensor:
    """
    Non-Maximum Suppression to remove duplicate detections.
    
    Args:
        boxes: Tensor of shape (N, 4) with format (x, y, w, h) normalized
        scores: Tensor of shape (N,) with confidence scores
        iou_threshold: IoU threshold for NMS
    
    Returns:
        Indices of boxes to keep
    """
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long)
    
    # Sort by scores descending
    _, indices = scores.sort(descending=True)
    keep = []
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current.item())
        
        if len(indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        current_box = boxes[current:current+1]
        remaining_boxes = boxes[indices[1:]]
        ious = calculate_iou(current_box, remaining_boxes).squeeze(0)
        
        # Keep boxes with IoU < threshold
        mask = ious < iou_threshold
        indices = indices[1:][mask]
    
    return torch.tensor(keep, dtype=torch.long)


def denormalize_boxes(boxes: torch.Tensor, img_width: int, img_height: int) -> torch.Tensor:
    """
    Convert normalized boxes (x, y, w, h) to pixel coordinates.
    
    Args:
        boxes: Tensor of shape (N, 4) with normalized coordinates [0, 1]
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        Tensor of shape (N, 4) with pixel coordinates (x, y, w, h)
    """
    denorm_boxes = boxes.clone()
    denorm_boxes[:, 0] *= img_width   # x
    denorm_boxes[:, 1] *= img_height  # y
    denorm_boxes[:, 2] *= img_width   # w
    denorm_boxes[:, 3] *= img_height  # h
    return denorm_boxes


def visualize_detections(
    image: np.ndarray,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: Optional[torch.Tensor] = None,
    class_names: List[str] = None,
    save_path: Optional[str] = None
):
    """
    Visualize object detections on an image.
    
    Args:
        image: Image as numpy array (H, W, 3) in RGB format
        boxes: Tensor of shape (N, 4) with format (x, y, w, h) in pixel coordinates
        labels: Tensor of shape (N,) with class indices
        scores: Optional tensor of shape (N,) with confidence scores
        class_names: List of class names (e.g., ['person', 'car', 'dog'])
        save_path: Optional path to save the visualization
    """
    if class_names is None:
        class_names = ['person', 'car', 'dog']
    
    # Convert to numpy if tensor
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if scores is not None and isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    colors = ['red', 'blue', 'green']
    
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x, y, w, h = box
        x1, y1 = x - w/2, y - h/2
        
        # Draw bounding box
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2,
            edgecolor=colors[int(label) % len(colors)],
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        label_text = class_names[int(label)]
        if scores is not None:
            label_text += f' {scores[i]:.2f}'
        
        ax.text(
            x1, y1 - 5,
            label_text,
            color=colors[int(label) % len(colors)],
            fontsize=10,
            weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
        )
    
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()
    
    plt.close()

