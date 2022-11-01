import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b1, efficientnet_b4, efficientnet_v2_l
import torchvision.models as models
from facenet_pytorch import InceptionResnetV1
import torch


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


class EfficientNet_B1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.8, inplace=True), nn.Linear(1280, num_classes, bias=True)
        )

        self.name = "EfficientNet_B1"

        self.init_params()

    def forward(self, x):
        x = self.model(x)
        return x

    def init_params(self):
        nn.init.kaiming_uniform_(self.model.classifier[1].weight)
        nn.init.zeros_(self.model.classifier[1].bias)


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


class InceptionResnet_MS(nn.Module):
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

    def __init__(self, num_classes, classifier_num, dropout_p):
        super().__init__()
        self.backbone = InceptionResnetV1(pretrained="vggface2", classify=True,)
        self.n_features = self.backbone.logits.out_features
        self.classifier = nn.Linear(self.n_features, num_classes)  # no dropout

        self.classifier_num = classifier_num
        self.dropout_p = dropout_p
        self.high_dropout = nn.Dropout(p=self.dropout_p)

        self.init_weights(self.classifier)

    def forward(self, x):
        x = self.backbone(x)
        logits = torch.mean(
            torch.stack(
                [self.classifier(self.high_dropout(x)) for _ in range(self.classifier_num)], dim=0,
            ),
            dim=0,
        )
        # x = logits
        return logits

    def init_weights(self, m):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)
