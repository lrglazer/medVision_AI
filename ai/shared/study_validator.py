from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def find_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def create_balanced_records(
    positive_paths: Iterable[Path],
    negative_paths: Iterable[Path],
    seed: int = 42,
) -> list[tuple[Path, int]]:
    positives = list(positive_paths)
    negatives = list(negative_paths)

    if not positives:
        raise RuntimeError("No positive validator images were found.")
    if not negatives:
        raise RuntimeError("No negative validator images were found.")

    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    sample_count = min(len(positives), len(negatives))
    records = [(path, 1) for path in positives[:sample_count]]
    records += [(path, 0) for path in negatives[:sample_count]]
    rng.shuffle(records)
    return records


class BinaryImageDataset(Dataset):
    def __init__(
        self,
        records: list[tuple[Path, int]],
        image_size: int,
        training: bool,
    ) -> None:
        self.records = records

        if training:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(6),
                    transforms.ColorJitter(
                        brightness=0.10,
                        contrast=0.10,
                        saturation=0.10,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        path, label = self.records[index]

        try:
            image = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError) as error:
            raise RuntimeError(f"Could not read image: {path}") from error

        return (
            self.transform(image),
            torch.tensor(label, dtype=torch.float32),
        )


def build_validator_model(pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    input_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(input_features, 1)
    return model
