import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b1, efficientnet_b4, efficientnet_v2_l
import torchvision.models as models
from facenet_pytorch import InceptionResnetV1


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

class Efficientnet_b1_mh(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = efficientnet_b1(weight="DEFAULT")
        self.n_features = self.backbone.classifier[1].in_features
        self.backbone = self.backbone.features
        self.backbone = nn.Sequential(*self.backbone)

        self.mask_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(0.2),
            nn.Conv2d(self.n_features, self.n_features, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.n_features),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(self.n_features, 3, 1, 1, 0, bias=True)
        )

        self.gender_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(0.2),
            nn.Conv2d(self.n_features, self.n_features, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.n_features),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(self.n_features, 2, 1, 1, 0, bias=True)
        )

        self.age_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(0.2),
            nn.Conv2d(self.n_features, self.n_features, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.n_features),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(self.n_features, 3, 1, 1, 0, bias=True)
        )

        self.init_weights()

    def forward(self, x):
        x = self.backbone(x)

        mask = self.mask_classifier(x)
        gender = self.gender_classifier(x)
        age = self.age_classifier(x)

        return mask, gender, age

    def init_weights(self):
        for m in self._modules:
            if isinstance(m, nn.Conv2d):
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
