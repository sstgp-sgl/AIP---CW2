"""Classification loss plus BCE and soft Dice on the lesion map.

Total = class-weighted cross-entropy
      + lambda_bce * binary cross-entropy
      + lambda_dice * (1 - soft Dice).

Dice is computed per image, then averaged over the batch. A smooth term of 1
keeps the score defined when a crop contains little or no lesion. The mask
terms do not choose a disease class.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationPlusBceDiceLoss(nn.Module):
    def __init__(
        self,
        class_weight: torch.Tensor,
        label_smoothing: float = 0.05,
        lambda_bce: float = 1.0,
        lambda_dice: float = 1.0,
        dice_smooth: float = 1.0,
    ):
        super().__init__()
        self.classification = nn.CrossEntropyLoss(
            weight=class_weight, label_smoothing=label_smoothing
        )
        self.bce = nn.BCEWithLogitsLoss()
        self.lambda_bce = float(lambda_bce)
        self.lambda_dice = float(lambda_dice)
        self.dice_smooth = float(dice_smooth)

    def soft_dice_loss(
        self, mask_logits: torch.Tensor, mask_target: torch.Tensor
    ) -> torch.Tensor:
        probs = torch.sigmoid(mask_logits)
        dims = tuple(range(1, probs.dim()))
        intersection = (probs * mask_target).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + mask_target.sum(dim=dims)
        dice = (2.0 * intersection + self.dice_smooth) / (
            cardinality + self.dice_smooth
        )
        return (1.0 - dice).mean()

    def forward(
        self,
        class_logits: torch.Tensor,
        labels: torch.Tensor,
        mask_logits: torch.Tensor,
        mask_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        class_loss = self.classification(class_logits, labels)
        bce_loss = self.bce(mask_logits, mask_target)
        dice_loss = self.soft_dice_loss(mask_logits, mask_target)
        total = (
            class_loss
            + self.lambda_bce * bce_loss
            + self.lambda_dice * dice_loss
        )
        return total, class_loss, bce_loss, dice_loss
