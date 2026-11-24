"""One training photo paired with its binary lesion mask.

Crops, flips and rotations are drawn once and applied to both. The photo uses
the baseline interpolation; the mask uses nearest-neighbour and stays binary.
Colour jitter and ImageNet normalisation touch only the photo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import functional as image_ops
from torchvision.transforms import transforms
from torchvision.transforms.functional import InterpolationMode

from baseline.data import IMAGENET_MEAN, IMAGENET_STD


def binary_lesion_mask(path: Path) -> Image.Image:
    """Background is 0. Any stored lesion code becomes 255."""
    with Image.open(path) as source:
        values = np.array(source)
    if values.ndim == 3:
        lesion = np.any(values > 0, axis=2)
    else:
        lesion = values > 0
    return Image.fromarray(lesion.astype(np.uint8) * 255, mode="L")


class SharedGeometry:
    """Baseline train geometry, with the mask carried along."""

    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.crop = transforms.RandomResizedCrop(
            image_size, scale=(0.8, 1.0), ratio=(3 / 4, 4 / 3)
        )
        self.rotation = transforms.RandomRotation(10)
        self.colour = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        )

    def __call__(self, image: Image.Image, mask: Image.Image):
        top, left, height, width = self.crop.get_params(
            image, self.crop.scale, self.crop.ratio
        )
        size = [self.image_size, self.image_size]
        image = image_ops.resized_crop(
            image,
            top,
            left,
            height,
            width,
            size,
            self.crop.interpolation,
            antialias=self.crop.antialias,
        )
        mask = image_ops.resized_crop(
            mask,
            top,
            left,
            height,
            width,
            size,
            InterpolationMode.NEAREST,
        )

        if torch.rand(1) < 0.5:
            image = image_ops.hflip(image)
            mask = image_ops.hflip(mask)
        if torch.rand(1) < 0.5:
            image = image_ops.vflip(image)
            mask = image_ops.vflip(mask)

        angle = self.rotation.get_params(self.rotation.degrees)
        image = image_ops.rotate(
            image,
            angle,
            interpolation=self.rotation.interpolation,
            expand=self.rotation.expand,
            center=self.rotation.center,
            fill=self.rotation.fill,
        )
        mask = image_ops.rotate(
            mask,
            angle,
            interpolation=InterpolationMode.NEAREST,
            expand=False,
            fill=0,
        )

        image = self.colour(image)
        image = image_ops.normalize(
            image_ops.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD
        )
        lesion = image_ops.to_tensor(mask) > 0
        return image, lesion.to(dtype=torch.float32)


class ImageMaskDataset(Dataset):
    """Known training rows only. Each item is one photo and one lesion mask."""

    def __init__(self, rows, data_root: str | Path, image_size: int = 224):
        self.rows = rows.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.geometry = SharedGeometry(image_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows.iloc[index]
        image_path = self.data_root / str(row["image_path"])
        mask_path = self.data_root / str(row["mask_path"])
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)

        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        mask = binary_lesion_mask(mask_path)
        image, mask = self.geometry(image, mask)
        return {
            "image": image,
            "mask": mask,
            "label": int(row["label"]),
            "image_id": str(row["image_id"]),
        }
