"""ResNet-18 whose class scores use a predicted lesion map.

The decoder matches the 7x7 lesion decoder. Its 7x7 logits are sigmoided and
used before global pooling:

    attended = features + features * sigmoid(lesion logits)

The linear head is unchanged. Calling the module with images only still
returns class logits, and those logits depend on the predicted map.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class SpatialLesionDecoder(nn.Module):
    """Same two convolutions as the 7x7 decoder. Returns the 7x7 map."""

    def __init__(self, in_channels: int = 512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class LesionAwareResNet18(nn.Module):
    def __init__(
        self,
        num_classes: int = 25,
        dropout: float = 0.2,
        pretrained: bool = True,
    ):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
        self.decoder = SpatialLesionDecoder(in_features)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(images)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x

    def forward(
        self, images: torch.Tensor, return_mask: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(images)
        lesion_logits = self.decoder(features)
        weight = torch.sigmoid(lesion_logits)
        attended = features + features * weight
        pooled = torch.flatten(self.backbone.avgpool(attended), 1)
        logits = self.head(pooled)
        if not return_mask:
            return logits
        mask_logits = F.interpolate(
            lesion_logits,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return logits, mask_logits
