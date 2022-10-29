import torch
import torch.nn as nn
from torchvision import models



class EfficientnetB0(nn.Module):
    def __init__(self, num_classes=18):
        super(EfficientnetB0, self).__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.fc = nn.Linear(1000, num_classes)
        self.name = 'Efficientnet-b0'
        
        self.init_params()
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x
    
    def init_params(self):
        nn.init.kaiming_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    
class EfficientnetB1(nn.Module):
    def __init__(self, num_classes=18):
        super(EfficientnetB1, self).__init__()
        self.backbone = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        self.fc = nn.Linear(1000, num_classes)
        
        self.name = "Efficientnet-b1"
        self.init_params()
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x
    
    def init_params(self):
        nn.init.kaiming_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        

class EfficientnetB2(nn.Module):
    def __init__(self, num_classes=18):
        super(EfficientnetB2, self).__init__()
        self.backbone = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        self.fc = nn.Linear(1000, num_classes)
        
        self.name = "Efficientnet-b2"
        self.init_params()
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x
    
    def init_params(self):
        nn.init.kaiming_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    
class EfficientnetB3(nn.Module):
    def __init__(self, num_classes=18):
        super(EfficientnetB3, self).__init__()
        self.backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        self.fc = nn.Linear(1000, num_classes)
        
        self.name = "Efficientnet-b3"
        self.init_params()
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x
    
    def init_params(self):
        nn.init.kaiming_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    
    
    
if __name__ == '__main__':
    model = EfficientnetB0()
    x = torch.randn(1, 3, 384, 512)
    y = model(x)
    
    model = EfficientnetB3()
    print(model.parameters)