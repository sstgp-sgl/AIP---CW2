"""ResNet-18 classification baseline (no lesion decoder)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class BaselineResNet18(nn.Module):
    """ResNet-18 with its ImageNet FC replaced by dropout + a 25-class head."""

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

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(images))
