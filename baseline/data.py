"""Manifest-backed RGB dataset for the baseline.

The baseline uses known training images only, with ImageNet preprocessing.
Validation and test loaders keep known and unknown rows because the evaluator
needs both; they never load masks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["included"] = frame["included"].astype(bool)
    frame["known_class_id"] = pd.to_numeric(
        frame["known_class_id"], errors="coerce"
    )
    return frame


def split_rows(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    """Return included rows for one experimental split.

    Training contains known classes only. Validation and test keep known plus
    their assigned unknown classes; unknown labels are -1.
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split: {split}")

    rows = frame[
        frame["included"] & (frame["experimental_split"] == split)
    ].copy()
    if split == "train":
        rows = rows[rows["class_role"] == "known"]

    rows["is_unknown"] = rows["class_role"] != "known"
    rows["label"] = np.where(
        rows["is_unknown"], -1, rows["known_class_id"]
    ).astype(int)
    return rows.reset_index(drop=True)


def class_names(train_rows: pd.DataFrame) -> list[str]:
    """Known class names ordered by known_class_id."""
    classes = (
        train_rows[["known_class_id", "host", "disease"]]
        .drop_duplicates("known_class_id")
        .sort_values("known_class_id")
    )
    return (
        classes["host"].astype(str) + " :: " + classes["disease"].astype(str)
    ).tolist()


def class_weights(train_rows: pd.DataFrame, num_classes: int) -> torch.Tensor:
    """Inverse-square-root class weights, normalised to mean 1."""
    counts = np.bincount(
        train_rows["label"].astype(int), minlength=num_classes
    )
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class BaselineDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, data_root: str | Path, transform):
        self.rows = rows.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows.iloc[index]
        with Image.open(self.data_root / str(row["image_path"])) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")

        shared = row["disease_shared_with_known"]
        shared = (
            False
            if pd.isna(shared) or str(shared).strip() == ""
            else str(shared).strip().lower() == "true"
        )

        return {
            "image": self.transform(image),
            "label": int(row["label"]),
            "is_unknown": bool(row["is_unknown"]),
            "unknown_group": str(row["unknown_group"]),
            "disease_shared_with_known": bool(shared),
            "image_id": str(row["image_id"]),
        }


def train_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size, scale=(0.8, 1.0), ratio=(3 / 4, 4 / 3)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
