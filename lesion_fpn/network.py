"""ResNet-18 with the baseline classification head and an FPN lesion decoder.

Classification still pools the last 7x7 map. The decoder builds a feature
pyramid from layer1 through layer4, so the lesion map is predicted from a
56x56 map instead of 7x7. Evaluation uses classification logits only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class FeaturePyramid(nn.Module):
    """Top-down pyramid. Returns the finest map, at layer1 resolution."""

    def __init__(self, in_channels: tuple[int, ...] = (64, 128, 256, 512), out_channels: int = 128):
        super().__init__()
        self.lateral = nn.ModuleList(
            nn.Conv2d(channels, out_channels, kernel_size=1) for channels in in_channels
        )
        self.smooth = nn.ModuleList(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        laterals = [conv(feature) for conv, feature in zip(self.lateral, features)]
        for index in range(len(laterals) - 1, 0, -1):
            laterals[index - 1] = laterals[index - 1] + F.interpolate(
                laterals[index],
                size=laterals[index - 1].shape[-2:],
                mode="nearest",
            )
        finest = self.smooth[0](laterals[0])
        return finest


class FpnLesionDecoder(nn.Module):
    def __init__(self, channels: int = 128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

    def forward(self, features: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        logits = self.layers(features)
        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)


class FpnLesionResNet18(nn.Module):
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
        self.pyramid = FeaturePyramid()
        self.decoder = FpnLesionDecoder()

    def stages(self, images: torch.Tensor) -> list[torch.Tensor]:
        x = self.backbone.conv1(images)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        c2 = self.backbone.layer1(x)
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)
        return [c2, c3, c4, c5]

    def forward(
        self, images: torch.Tensor, return_mask: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.stages(images)
        pooled = torch.flatten(self.backbone.avgpool(features[-1]), 1)
        logits = self.head(pooled)
        if not return_mask:
            return logits
        mask_logits = self.decoder(self.pyramid(features), images.shape[-2:])
        return logits, mask_logits
