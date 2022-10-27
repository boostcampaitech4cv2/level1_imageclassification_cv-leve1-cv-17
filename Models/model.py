import torch
import torch.nn as nn
from torchvision import models



class EfficientnetB0(nn.Module):
    def __init__(self, num_classes=18):
        super(EfficientnetB0, self).__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.fc = nn.Linear(1000, num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x
    
    
if __name__ == '__main__':
    model = EfficientnetB0()
    x = torch.randn(1, 3, 384, 512)
    y = model(x)
    print(y)