from torchvision import models
import torch.nn as nn

class EfficientNet_b1(nn.Module):
    def __init__(self, num_classes=18):
        super(EfficientNet_b1, self).__init__()
        self.backbone = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        self.classifier = nn.Linear(1000, num_classes)
        self.name = "EfficientNet_b1"

        self.init_params()
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    def init_params(self):
        nn.init.kaiming_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)