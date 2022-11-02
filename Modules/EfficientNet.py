from torchvision import models
import torch.nn as nn

class EfficientNet_b0(nn.Module):
    def __init__(self, num_classes=18):
        super(EfficientNet_b0, self).__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.classifier = nn.Linear(1000, num_classes)
        self.name = "EfficientNet_b0"

        self.init_params()
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    def init_params(self):
        nn.init.kaiming_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

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

class EfficientNet_b0_mh(nn.Module):
    def __init__(self, mask_classes=3, age_gender_classes=6):
        super(EfficientNet_b0_mh, self).__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.mask_classifier = nn.Linear(1000, mask_classes)
        self.age_gender_classifier = nn.Linear(1000, age_gender_classes)
        self.name = "EfficientNet_b0_mh"

        self.init_params()
        
    def forward(self, x):
        x = self.backbone(x)
        mask = self.mask_classifier(x)
        age_gender = self.age_gender_classifier(x)
        
        return mask, age_gender

    def init_params(self):
        nn.init.kaiming_uniform_(self.mask_classifier.weight)
        nn.init.zeros_(self.mask_classifier.bias)
        nn.init.kaiming_uniform_(self.age_gender_classifier.weight)
        nn.init.zeros_(self.age_gender_classifier.bias)

class EfficientNet_b0_3mh(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
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
            nn.Conv2d(self.n_features, 6, 1, 1, 0, bias=True)
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


class EfficientNet_b1_mh(nn.Module):
    def __init__(self, mask_classes=3, age_gender_classes=6):
        super(EfficientNet_b1_mh, self).__init__()
        self.backbone = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        self.mask_classifier = nn.Linear(1000, mask_classes)
        self.age_gender_classifier = nn.Linear(1000, age_gender_classes)
        self.name = "EfficientNet_b1_mh"

        self.init_params()
        
    def forward(self, x):
        x = self.backbone(x)
        mask = self.mask_classifier(x)
        age_gender = self.age_gender_classifier(x)
        
        return mask, age_gender

    def init_params(self):
        nn.init.kaiming_uniform_(self.mask_classifier.weight)
        nn.init.zeros_(self.mask_classifier.bias)
        nn.init.kaiming_uniform_(self.age_gender_classifier.weight)
        nn.init.zeros_(self.age_gender_classifier.bias)

class EfficientNet_b2_mh(nn.Module):
    def __init__(self, mask_classes=3, age_gender_classes=6):
        super(EfficientNet_b2_mh, self).__init__()
        self.backbone = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        self.mask_classifier = nn.Linear(1000, mask_classes)
        self.age_gender_classifier = nn.Linear(1000, age_gender_classes)
        self.name = "EfficientNet_b2_mh"

        self.init_params()
        
    def forward(self, x):
        x = self.backbone(x)
        mask = self.mask_classifier(x)
        age_gender = self.age_gender_classifier(x)
        
        return mask, age_gender

    def init_params(self):
        nn.init.kaiming_uniform_(self.mask_classifier.weight)
        nn.init.zeros_(self.mask_classifier.bias)
        nn.init.kaiming_uniform_(self.age_gender_classifier.weight)
        nn.init.zeros_(self.age_gender_classifier.bias)

_model_entrypoints = {
    'EfficientNet_b0': EfficientNet_b0,
    'EfficientNet_b1': EfficientNet_b1,
    'EfficientNet_b2': EfficientNet_b2,
    'EfficientNet_b0_mh': EfficientNet_b0_mh,
    'EfficientNet_b1_mh': EfficientNet_b1_mh,
    'EfficientNet_b2_mh': EfficientNet_b2_mh,
    'EfficientNet_b0_3mh': EfficientNet_b0_3mh
}

def model_entrypoint(model_name):
    return _model_entrypoints[model_name]

def in_entrypoints(model_name):
    return model_name in _model_entrypoints

def create_model(model_name, **kwargs):
    if in_entrypoints(model_name):
        create_fn = model_entrypoint(model_name)
        model = create_fn(**kwargs)
    else:
        raise RuntimeError('Unknown model (%s)' % model_name)
    return model
