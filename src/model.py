"""
Custom CNN object detector model from scratch
"""
import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Tuple


class ObjectDetector(nn.Module):
    
    def __init__(self, num_classes: int = 3, input_size: int = 224):
        
        super(ObjectDetector, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        feature_size = input_size // 8
        feature_dim = 256 * feature_size * feature_size
        
        self.feature_fc = nn.Linear(feature_dim, 512)
        self.feature_relu = nn.ReLU()
        
        self.max_detections = 10  
        self.bbox_head = nn.Linear(512, self.max_detections * 4)
        
        self.class_head = nn.Linear(512, self.max_detections * (num_classes + 1))
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in [self.conv1, self.conv2, self.conv3]:
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        
        for m in [self.feature_fc, self.bbox_head, self.class_head]:
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size = x.size(0)
        
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        
        x = x.view(batch_size, -1)
        
        features = self.feature_relu(self.feature_fc(x))
        
        bbox_output = self.bbox_head(features)
        class_output = self.class_head(features)
        
        bboxes = bbox_output.view(batch_size, self.max_detections, 4)
        bboxes = torch.sigmoid(bboxes)
        
        class_scores = class_output.view(batch_size, self.max_detections, self.num_classes + 1)
        
        return bboxes, class_scores


def create_model(num_classes: int = 3, input_size: int = 224) -> ObjectDetector:
    
    model = ObjectDetector(num_classes=num_classes, input_size=input_size)
    return model

