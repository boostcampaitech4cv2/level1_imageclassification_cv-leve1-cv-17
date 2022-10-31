import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b1, efficientnet_b4, efficientnet_v2_l
import torchvision.models as models
from facenet_pytorch import InceptionResnetV1
import timm


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
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = efficientnet_b1(pretrained=True)
        self.n_features = self.backbone.classifier[1].out_features
        self.classifier = nn.Linear(self.n_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


class Efficientnet_b1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = efficientnet_b1(weight="DEFAULT")
        self.n_features = self.backbone.classifier[1].out_features
        self.classifier = nn.Linear(self.n_features, num_classes)

        self.init_weights(self.classifier)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    def init_weights(self, m):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)


class InceptionResnet(nn.Module):
    """
    Total params: 27,979,383
    Trainable params: 27,979,383
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 2.25
    Forward/backward pass size (MB): 840.53
    Params size (MB): 106.73
    Estimated Total Size (MB): 949.52
    """

    def __init__(self, num_classes):
        super().__init__()
        self.backbone = InceptionResnetV1(pretrained="vggface2", classify=True,)
        self.n_features = self.backbone.logits.out_features
        self.classifier = nn.Linear(self.n_features, num_classes)

        self.init_weights(self.classifier)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    def init_weights(self, m):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)


class Efficientnet_v2_l(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = efficientnet_v2_l(weight="DEFAULT")
        self.n_features = self.backbone.classifier[1].out_features
        self.classifier = nn.Linear(self.n_features, num_classes)

        self.init_weights(self.classifier)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    def init_weights(self, m):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)
        
        
'''
https://github.com/facebookresearch/pycls/issues/78
RegNetY models come from a design space that also includes the Squeeze-and-Excitation (SE) operation 
(RegNetY = RegNetX + SE).

https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/regnet.py
RegNet models 
- regnetx: regnetx_002, 004, 006, 008, 016, 032, 040, 064, 080, 120, 160, 320
- regnety: regnety_002, 004, 006, 008, 016, 032, 040, 064, 080, 120, 160, 320
'''
class RegnetX002(nn.Module):
    def __init__(self, num_classes=18):
        super(RegnetX002, self).__init__()
        self.backbone = timm.create_model('regnetx_002', pretrained=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1000, num_classes)
        )
        
        
        self.name = "RegnetX002"
        self.init_params()
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x
    
    def init_params(self):
        nn.init.kaiming_uniform_(self.fc[1].weight)
        nn.init.zeros_(self.fc[1].bias)
        
        
if __name__ == '__main__':
    model = RegnetX002()
    print(model)