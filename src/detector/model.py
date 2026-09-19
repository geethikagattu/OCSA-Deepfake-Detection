import torch
import torch.nn as nn
from torchvision import models


class DeepfakeDetector(nn.Module):
    """
    Simple baseline deepfake detector.

    Input:
        RGB face/image tensor

    Output:
        Logit representing probability of:
        0 = Real
        1 = Fake
    """

    def __init__(self, pretrained=True):
        super().__init__()

        weights = (
            models.ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        self.backbone = models.resnet18(weights=weights)

        # Replace ImageNet classifier with binary classifier.
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.backbone(x).squeeze(1)
