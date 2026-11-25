"""ResNet-18 with a baseline global head and a lesion-only head.

The lesion map is predicted from the last 7x7 features. The lesion vector is
the weighted average of those features, with weights from the sigmoid of the
7x7 lesion logits. Ground-truth masks supervise only the upsampled map.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class SpatialLesionDecoder(nn.Module):
    def __init__(self, in_channels: int = 512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class ConsistencyResNet18(nn.Module):
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
        self.global_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
        self.lesion_head = nn.Sequential(
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

    def lesion_vector(
        self, features: torch.Tensor, lesion_logits: torch.Tensor
    ) -> torch.Tensor:
        weight = torch.sigmoid(lesion_logits)
        denominator = weight.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        pooled = (features * weight).sum(dim=(2, 3), keepdim=True) / denominator
        return torch.flatten(pooled, 1)

    def forward(
        self, images: torch.Tensor, return_all: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encode(images)
        global_logits = self.global_head(
            torch.flatten(self.backbone.avgpool(features), 1)
        )
        if not return_all:
            return global_logits
        lesion_logits_map = self.decoder(features)
        lesion_logits = self.lesion_head(
            self.lesion_vector(features, lesion_logits_map)
        )
        mask_logits = F.interpolate(
            lesion_logits_map,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return global_logits, lesion_logits, mask_logits
