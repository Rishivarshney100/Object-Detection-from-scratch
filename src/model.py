"""
Custom CNN object detector model from scratch.
"""
import torch
import torch.nn as nn
import torch.nn.init as init


class ObjectDetector(nn.Module):
    """
    Custom CNN-based object detector trained from scratch.
    
    Architecture:
    - 3 Convolution blocks (Conv → BatchNorm → ReLU → MaxPool)
    - Feature extraction
    - Two detection heads: Bounding box regression and classification
    """
    
    def __init__(self, num_classes: int = 3, input_size: int = 224):
        """
        Initialize the object detector.
        
        Args:
            num_classes: Number of object classes (excluding background)
            input_size: Input image size (assumed square)
        """
        super(ObjectDetector, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Backbone: 3 Convolution blocks
        # Block 1: 3 → 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2: 64 → 128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3: 128 → 256
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate feature map size after 3 pooling layers (each reduces by 2x)
        # input_size / 2^3 = input_size / 8
        feature_size = input_size // 8
        feature_dim = 256 * feature_size * feature_size
        
        # Feature extraction layer
        self.feature_fc = nn.Linear(feature_dim, 512)
        self.feature_relu = nn.ReLU()
        
        # Detection heads
        # Bounding box head: outputs 4 values (x, y, w, h) per object
        # Note: For simplicity, we output a fixed number of detections
        # In practice, you might want to use anchor-based or anchor-free approach
        self.max_detections = 10  # Maximum number of detections per image
        self.bbox_head = nn.Linear(512, self.max_detections * 4)
        
        # Classification head: outputs (num_classes + 1) scores per detection
        # +1 for background class
        self.class_head = nn.Linear(512, self.max_detections * (num_classes + 1))
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He (conv) and Xavier (FC) initialization."""
        # He initialization for convolutional layers
        for m in [self.conv1, self.conv2, self.conv3]:
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        
        # Xavier initialization for fully connected layers
        for m in [self.feature_fc, self.bbox_head, self.class_head]:
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            bboxes: Tensor of shape (B, max_detections, 4) with (x, y, w, h) normalized
            class_scores: Tensor of shape (B, max_detections, num_classes + 1) with logits
        """
        batch_size = x.size(0)
        
        # Backbone
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(batch_size, -1)
        
        # Feature extraction
        features = self.feature_relu(self.feature_fc(x))
        
        # Detection heads
        bbox_output = self.bbox_head(features)
        class_output = self.class_head(features)
        
        # Reshape outputs
        bboxes = bbox_output.view(batch_size, self.max_detections, 4)
        # Apply sigmoid to normalize bbox coordinates to [0, 1]
        bboxes = torch.sigmoid(bboxes)
        
        class_scores = class_output.view(batch_size, self.max_detections, self.num_classes + 1)
        
        return bboxes, class_scores


def create_model(num_classes: int = 3, input_size: int = 224) -> ObjectDetector:
    """
    Create and return an ObjectDetector model.
    
    Args:
        num_classes: Number of object classes
        input_size: Input image size
    
    Returns:
        Initialized ObjectDetector model
    """
    model = ObjectDetector(num_classes=num_classes, input_size=input_size)
    return model

