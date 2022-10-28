import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class EfficientNet_B0(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B1_Weights.DEFAULT)
        self.classifier = nn.Linear(1000, num_classes)
        
        self.name = "EfficientNet_B0"

        self.init_params()
        
    def forward(self, x):
        x = self.backbone(x)
        x = F.relu(x)
        x = self.classifier(x)
        return x

    def init_params(self):
        nn.init.kaiming_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

class EfficientNet_B1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        self.classifier = nn.Linear(1000, num_classes)

        self.name = "EfficientNet_B1"

        self.init_params()

    def forward(self, x):
        x = self.backbone(x)
        x = F.relu(x)
        x = self.classifier(x)
        return x

    def init_params(self):
        nn.init.kaiming_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.classifier = nn.Linear(512, num_classes)

        self.name = "ResNet18"

        self.init_params()

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    def init_params(self):
        nn.init.kaiming_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

class ResNet34(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.classifier = nn.Linear(512, num_classes)

        self.name = "ResNet34"

        self.init_params()

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    def init_params(self):
        nn.init.kaiming_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.classifier = nn.Linear(512, num_classes)

        self.name = "ResNet50"

        self.init_params()

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    def init_params(self):
        nn.init.kaiming_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)