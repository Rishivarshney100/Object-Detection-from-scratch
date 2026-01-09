"""
Image inference script for object detection.
Detects objects in single images or directories of images.
"""
import torch
import torch.nn as nn
from pathlib import Path
import argparse
import numpy as np
import cv2
from PIL import Image

from src.model import create_model
from src.augmentation import get_val_augmentation
from src.utils import nms, denormalize_boxes


def preprocess_image(image: np.ndarray, input_size: int = 224) -> torch.Tensor:
    """Preprocess image for model input."""
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Apply augmentation (only resize and normalize)
    aug = get_val_augmentation(input_size)
    augmented = aug(image=image_rgb, bboxes=[], labels=[])
    image_tensor = augmented['image']
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


def detect_objects_in_image(
    model: nn.Module,
    image: np.ndarray,
    device: torch.device,
    input_size: int = 224,
    score_threshold: float = 0.15,
    nms_threshold: float = 0.4,
    debug: bool = False
) -> tuple:
    """
    Run object detection on an image.
    
    Returns:
        boxes: List of bounding boxes in pixel coordinates (x, y, w, h)
        labels: List of class labels
        scores: List of confidence scores
    """
    model.eval()
    original_height, original_width = image.shape[:2]
    
    # Preprocess
    image_tensor = preprocess_image(image, input_size)
    image_tensor = image_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        pred_bboxes, pred_classes = model(image_tensor)
    
    # Process predictions
    pred_bboxes = pred_bboxes[0].cpu()  # (max_detections, 4) normalized
    pred_scores = torch.softmax(pred_classes[0], dim=1).cpu()  # (max_detections, num_classes+1)
    
    # Get max class scores and indices (excluding background)
    max_scores, class_indices = torch.max(pred_scores[:, :3], dim=1)
    
    # Sort by score descending to prioritize high-confidence detections
    sorted_indices = torch.argsort(max_scores, descending=True)
    sorted_boxes = pred_bboxes[sorted_indices]
    sorted_scores = max_scores[sorted_indices]
    sorted_labels = class_indices[sorted_indices]
    
    # Filter by score threshold
    valid_mask = sorted_scores >= score_threshold
    if debug:
        print(f"  [Debug] Total predictions: {len(sorted_scores)}")
        print(f"  [Debug] Above threshold {score_threshold:.2f}: {valid_mask.sum().item()}")
        print(f"  [Debug] Top 10 scores: {sorted_scores[:10].tolist()}")
    
    if valid_mask.sum() == 0:
        # If no detections above threshold, show top 5 for debugging
        if debug:
            print(f"  [Debug] No detections above threshold {score_threshold:.2f}")
            print(f"  [Debug] Top 5 scores: {sorted_scores[:5].tolist()}")
        return [], [], []
    
    pred_boxes_filtered = sorted_boxes[valid_mask]
    pred_scores_filtered = sorted_scores[valid_mask]
    pred_labels_filtered = sorted_labels[valid_mask]
    
    # Apply NMS with lower threshold to allow more detections
    if len(pred_boxes_filtered) > 0:
        keep_indices = nms(pred_boxes_filtered, pred_scores_filtered, iou_threshold=nms_threshold)
        pred_boxes_filtered = pred_boxes_filtered[keep_indices]
        pred_scores_filtered = pred_scores_filtered[keep_indices]
        pred_labels_filtered = pred_labels_filtered[keep_indices]
    
    # Denormalize boxes to pixel coordinates
    pred_boxes_pixel = denormalize_boxes(pred_boxes_filtered, original_width, original_height)
    
    return pred_boxes_pixel, pred_labels_filtered, pred_scores_filtered


def draw_detections(
    image: np.ndarray,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    class_names: list
) -> np.ndarray:
    """Draw bounding boxes and labels on image."""
    if len(boxes) == 0:
        return image
    
    # Use bright lime-green color for bounding boxes (BGR format)
    box_color = (0, 255, 0)  # Bright lime-green
    
    for box, label, score in zip(boxes, labels, scores):
        # Get box coordinates (center format: x, y, w, h)
        if isinstance(box, torch.Tensor):
            x, y, w, h = box.cpu().numpy()
        else:
            x, y, w, h = box
        
        # Convert to corner format (x1, y1, x2, y2)
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        
        # Clamp to image bounds
        x1 = max(0, min(x1, image.shape[1] - 1))
        y1 = max(0, min(y1, image.shape[0] - 1))
        x2 = max(x1 + 1, min(x2, image.shape[1]))
        y2 = max(y1 + 1, min(y2, image.shape[0]))
        
        label_idx = label.item() if isinstance(label, torch.Tensor) else int(label)
        class_name = class_names[label_idx]
        score_val = score.item() if isinstance(score, torch.Tensor) else score
        
        # Draw bright lime-green bounding box with thicker line
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 3)
        
        # Prepare label text in format "Car : 97%"
        label_text = f"{class_name.capitalize()} : {int(score_val * 100)}%"
        
        # Get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, font, font_scale, thickness
        )
        
        # Calculate label position (above the box, slightly to the right of center)
        label_x = x1 + (x2 - x1) // 4  # Position slightly to the right
        label_y = y1 - 10
        
        # Ensure label doesn't go off screen
        if label_y < text_height + 10:
            label_y = y2 + text_height + 10  # Put below box if no room above
        
        # Draw black rectangular background for label
        padding = 8
        bg_x1 = label_x - padding
        bg_y1 = label_y - text_height - padding
        bg_x2 = label_x + text_width + padding
        bg_y2 = label_y + baseline + padding
        
        # Ensure background stays within image bounds
        bg_x1 = max(0, bg_x1)
        bg_y1 = max(0, bg_y1)
        bg_x2 = min(image.shape[1], bg_x2)
        bg_y2 = min(image.shape[0], bg_y2)
        
        # Draw solid black background
        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        
        # Draw white text on black background
        cv2.putText(
            image,
            label_text,
            (label_x, label_y),
            font,
            font_scale,
            (255, 255, 255),  # White text
            thickness
        )
    
    return image


def process_image(
    image_path: Path,
    model: nn.Module,
    device: torch.device,
    output_dir: Path,
    input_size: int = 224,
    score_threshold: float = 0.15,
    nms_threshold: float = 0.4,
    class_names: list = None,
    debug: bool = False
):
    """Process a single image and save result."""
    if class_names is None:
        class_names = ['person', 'car', 'dog']
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    original_image = image.copy()
    
    # Run detection
    boxes, labels, scores = detect_objects_in_image(
        model, image, device, input_size, score_threshold, nms_threshold, debug
    )
    
    # Draw detections
    result_image = draw_detections(original_image, boxes, labels, scores, class_names)
    
    # Save result
    output_path = output_dir / f"detected_{image_path.name}"
    cv2.imwrite(str(output_path), result_image)
    
    # Print results
    print(f"\n{image_path.name}:")
    if len(boxes) > 0:
        for box, label, score in zip(boxes, labels, scores):
            label_idx = label.item() if isinstance(label, torch.Tensor) else int(label)
            class_name = class_names[label_idx]
            score_val = score.item() if isinstance(score, torch.Tensor) else score
            print(f"  - {class_name.upper()}: {score_val:.1%}")
    else:
        print("  - No detections")
    print(f"  - Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Object detection on images')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to image file or directory')
    parser.add_argument('--input_size', type=int, default=224,
                        help='Input image size (default: 224)')
    parser.add_argument('--score_threshold', type=float, default=0.15,
                        help='Score threshold for detections (default: 0.15, lower = more detections)')
    parser.add_argument('--nms_threshold', type=float, default=0.4,
                        help='NMS IoU threshold (default: 0.4, lower = less aggressive suppression)')
    parser.add_argument('--debug', action='store_true',
                        help='Show debug information about detections')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results (default: same as input or sample_images/output)')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    print(f'Loading model from: {args.model_path}')
    
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Try to infer input_size from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Check feature_fc input dimension to infer original input_size
    inferred_input_size = args.input_size
    if 'feature_fc.weight' in state_dict:
        feature_fc_input_dim = state_dict['feature_fc.weight'].shape[1]
        feature_size_squared = feature_fc_input_dim / 256
        if feature_size_squared > 0:
            inferred_input_size = int((feature_size_squared ** 0.5) * 8)
        
        if inferred_input_size != args.input_size:
            print(f'Warning: Checkpoint was saved with input_size={inferred_input_size}, but current model uses input_size={args.input_size}')
            print(f'Using inferred input_size={inferred_input_size}')
            args.input_size = inferred_input_size
    
    model = create_model(num_classes=3, input_size=args.input_size)
    
    # Load checkpoint with strict=False to handle any remaining mismatches
    if 'model_state_dict' in checkpoint:
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    
    if missing_keys:
        print(f'Warning: Missing keys in checkpoint: {missing_keys}')
    if unexpected_keys:
        print(f'Warning: Unexpected keys in checkpoint: {unexpected_keys}')
    
    model = model.to(device)
    model.eval()
    
    class_names = ['person', 'car', 'dog']
    
    # Determine input path
    input_path = Path(args.image_path)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif input_path.is_dir():
        output_dir = input_path / 'output'
    else:
        output_dir = input_path.parent / 'output'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process images
    if input_path.is_file():
        # Single image
        print(f'\nProcessing single image: {input_path}')
        process_image(
            input_path, model, device, output_dir,
            args.input_size, args.score_threshold, args.nms_threshold, class_names, args.debug
        )
    elif input_path.is_dir():
        # Directory of images
        print(f'\nProcessing directory: {input_path}')
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions and not f.name.startswith('detected_')]
        
        if not image_files:
            print(f"No image files found in {input_path}")
            return
        
        print(f"Found {len(image_files)} images")
        for image_file in image_files:
            process_image(
                image_file, model, device, output_dir,
                args.input_size, args.score_threshold, args.nms_threshold, class_names, args.debug
            )
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        return
    
    print(f'\n\nAll results saved to: {output_dir}')


if __name__ == '__main__':
    main()
