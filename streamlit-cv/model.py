import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class EfficientnetB2_MD2(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.base_model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        self.base_model.classifier = Identity()

        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(16)])
        self.fc = nn.Linear(1408, num_classes)

        self.init_weights(self.fc)

    def forward(self, x):
        x = self.base_model(x)

        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = dropout(x.clone())
                out = self.fc(out)
            else:
                temp_out = dropout(x.clone())
                out += self.fc(temp_out)
        return torch.sigmoid(out/len(self.dropouts))
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)