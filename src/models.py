import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights



class SmallCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def build_resnet50(num_classes=2, pretrained=True):
    m = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def get_model(name="cnn_custom", num_classes=2, pretrained=True):
    name = name.lower()
    if name in ["cnn_custom","cnn","custom"]:
        return SmallCNN(num_classes)
    if name in ["resnet50","resnet"]:
        return build_resnet50(num_classes, pretrained)
    # ICI on ajoute EfficientNet
    if name in ["efficientnet_b0", "efficientnet", "effnet"]:
        return build_efficientnet_b0(num_classes, pretrained)
    raise ValueError(f"Unknown model: {name}")


def build_efficientnet_b0(num_classes=2, pretrained=True):
    m = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m
