"""ResNet-18 with the baseline classification head and a small lesion decoder.

The decoder reads the last spatial map, before global pooling. It has no skip
connections from earlier layers. Evaluation calls the module with images only
and receives classification logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class LesionDecoder(nn.Module):
    """Conv on the 512-channel map, then bilinear upsample to the photo size."""

    def __init__(self, in_channels: int = 512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
        )

    def forward(self, features: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        logits = self.layers(features)
        return F.interpolate(
            logits, size=out_size, mode="bilinear", align_corners=False
        )


class MaskResNet18(nn.Module):
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
        self.decoder = LesionDecoder(in_features)

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
        pooled = torch.flatten(self.backbone.avgpool(features), 1)
        logits = self.head(pooled)
        if not return_mask:
            return logits
        mask_logits = self.decoder(features, images.shape[-2:])
        return logits, mask_logits
