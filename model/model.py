import torch
import torch.nn as nn
import torchvision.models as models


class EfficientnetB0(nn.Module):
    def __init__(self, num_classes):
        super(EfficientnetB0, self).__init__()
        self.backbone = models.efficientnet_b0(pretrained=True)
        self.fc = nn.Linear(1000, num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x