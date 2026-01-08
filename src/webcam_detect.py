
import torch
import torch.nn as nn
from pathlib import Path
import argparse
import numpy as np
import cv2
from PIL import Image
import time

from src.model import create_model
from src.augmentation import get_val_augmentation
from src.utils import nms, denormalize_boxes


def preprocess_frame(frame: np.ndarray, input_size: int = 224) -> torch.Tensor:
    """Preprocess frame for model input."""
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Apply augmentation (only resize and normalize)
    aug = get_val_augmentation(input_size)
    augmented = aug(image=frame_rgb, bboxes=[], labels=[])
    image_tensor = augmented['image']
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


def detect_objects_in_frame(
    model: nn.Module,
    frame: np.ndarray,
    device: torch.device,
    input_size: int = 224,
    score_threshold: float = 0.3,
    nms_threshold: float = 0.5
) -> tuple:

    model.eval()
    original_height, original_width = frame.shape[:2]
    
    # Preprocess
    image_tensor = preprocess_frame(frame, input_size)
    image_tensor = image_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        pred_bboxes, pred_classes = model(image_tensor)
    
    # Process predictions
    pred_bboxes = pred_bboxes[0].cpu()  # (max_detections, 4) normalized
    pred_scores = torch.softmax(pred_classes[0], dim=1).cpu()  # (max_detections, num_classes+1)
    
    # Get max class scores and indices (excluding background)
    max_scores, class_indices = torch.max(pred_scores[:, :3], dim=1)
    
    # Filter by score threshold
    valid_mask = max_scores >= score_threshold
    if valid_mask.sum() == 0:
        return [], [], []
    
    pred_boxes_filtered = pred_bboxes[valid_mask]
    pred_scores_filtered = max_scores[valid_mask]
    pred_labels_filtered = class_indices[valid_mask]
    
    # Apply NMS
    if len(pred_boxes_filtered) > 0:
        keep_indices = nms(pred_boxes_filtered, pred_scores_filtered, iou_threshold=nms_threshold)
        pred_boxes_filtered = pred_boxes_filtered[keep_indices]
        pred_scores_filtered = pred_scores_filtered[keep_indices]
        pred_labels_filtered = pred_labels_filtered[keep_indices]
    
    # Denormalize boxes to pixel coordinates
    # Model outputs are normalized [0,1] based on input_size (224x224)
    # Since augmentation uses Resize (stretches to square), coordinates map directly to original dimensions
    pred_boxes_pixel = denormalize_boxes(pred_boxes_filtered, original_width, original_height)
    
    return pred_boxes_pixel, pred_labels_filtered, pred_scores_filtered


def draw_detections(
    frame: np.ndarray,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    class_names: list
) -> np.ndarray:
    if len(boxes) == 0:
        return frame
    
    # Use white color for all boxes
    box_color = (255, 255, 255)
    
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
        
        # Clamp to frame bounds (but don't restrict movement)
        x1 = max(0, min(x1, frame.shape[1] - 1))
        y1 = max(0, min(y1, frame.shape[0] - 1))
        x2 = max(x1 + 1, min(x2, frame.shape[1]))
        y2 = max(y1 + 1, min(y2, frame.shape[0]))
        
        label_idx = label.item() if isinstance(label, torch.Tensor) else int(label)
        class_name = class_names[label_idx]
        score_val = score.item() if isinstance(score, torch.Tensor) else score
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
        
        # Prepare label text - show class name prominently
        label_text = f"{class_name.upper()}"
        confidence_text = f"{score_val:.1%}"
        
        # Get text sizes
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3
        )
        (conf_width, conf_height), _ = cv2.getTextSize(
            confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        
        # Draw label background (black with transparency effect)
        label_bg_height = text_height + conf_height + 20
        label_bg_width = max(text_width, conf_width) + 20
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x1, y1 - label_bg_height),
            (x1 + label_bg_width, y1),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw class name (large, white, bold)
        cv2.putText(
            frame,
            label_text,
            (x1 + 10, y1 - conf_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3
        )
        
        # Draw confidence score (smaller, below class name)
        cv2.putText(
            frame,
            confidence_text,
            (x1 + 10, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2
        )
    
    return frame


def main():
    parser = argparse.ArgumentParser(description='Real-time object detection with webcam')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index (default: 0)')
    parser.add_argument('--input_size', type=int, default=224,
                        help='Input image size (default: 224)')
    parser.add_argument('--score_threshold', type=float, default=0.3,
                        help='Score threshold for detections (default: 0.3)')
    parser.add_argument('--nms_threshold', type=float, default=0.5,
                        help='NMS IoU threshold (default: 0.5)')
    parser.add_argument('--fps_display', action='store_true',
                        help='Display FPS on video')
    
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
        # feature_dim = 256 * (input_size // 8) * (input_size // 8)
        # So: (input_size // 8)^2 = feature_fc_input_dim / 256
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
    
    # Initialize webcam
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return
    
    # Set camera resolution (optional, adjust as needed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n" + "="*60)
    print("Object Detection Webcam - Live Detection")
    print("="*60)
    print("Detecting: person, car, dog")
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current frame")
    print("="*60 + "\n")
    
    class_names = ['person', 'car', 'dog']
    frame_count = 0
    fps_start_time = time.time()
    fps_counter = 0
    
    # Create output directory for saved frames
    output_dir = Path('webcam_outputs')
    output_dir.mkdir(exist_ok=True)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Run detection
            boxes, labels, scores = detect_objects_in_frame(
                model, frame, device, args.input_size,
                args.score_threshold, args.nms_threshold
            )
            
            # Draw detections
            frame = draw_detections(frame, boxes, labels, scores, class_names)
            
            # Display FPS
            if args.fps_display:
                fps_counter += 1
                if fps_counter >= 30:  # Update FPS every 30 frames
                    fps = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()
                else:
                    fps = 0
                
                if fps > 0:
                    cv2.putText(
                        frame,
                        f"FPS: {fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
            
            # Display detection count
            detection_text = f"Detections: {len(boxes)}"
            cv2.putText(
                frame,
                detection_text,
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Show frame
            cv2.imshow('Object Detection - Press q to quit', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = int(time.time())
                save_path = output_dir / f'detection_{timestamp}.jpg'
                cv2.imwrite(str(save_path), frame)
                print(f"Frame saved to: {save_path}")
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nProcessed {frame_count} frames")
        print(f"Saved frames are in: {output_dir}")


if __name__ == '__main__':
    main()

