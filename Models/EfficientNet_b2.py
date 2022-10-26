from torchvision import models
import torch.nn as nn

class EfficientNet_b2(nn.Module):
    def __init__(self, num_classes=18):
        super(EfficientNet_b2, self).__init__()
        self.backbone = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
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