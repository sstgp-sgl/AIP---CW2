"""Classification loss plus a weighted binary lesion loss.

Total = class-weighted cross-entropy + lambda_mask * binary cross-entropy.
The mask term compares the decoder map with the binary lesion mask. It does
not choose a disease class.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationPlusMaskLoss(nn.Module):
    def __init__(
        self,
        class_weight: torch.Tensor,
        label_smoothing: float = 0.05,
        lambda_mask: float = 1.0,
    ):
        super().__init__()
        self.classification = nn.CrossEntropyLoss(
            weight=class_weight, label_smoothing=label_smoothing
        )
        self.mask = nn.BCEWithLogitsLoss()
        self.lambda_mask = float(lambda_mask)

    def forward(
        self,
        class_logits: torch.Tensor,
        labels: torch.Tensor,
        mask_logits: torch.Tensor,
        mask_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        class_loss = self.classification(class_logits, labels)
        mask_loss = self.mask(mask_logits, mask_target)
        total = class_loss + self.lambda_mask * mask_loss
        return total, class_loss, mask_loss
