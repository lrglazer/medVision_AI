from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def infer_label(path_string: str) -> int:
    lowered = path_string.lower()

    if "positive" in lowered:
        return 1

    if "negative" in lowered:
        return 0

    raise ValueError(
        f"Could not infer MURA label from path: {path_string}"
    )


def infer_body_part(path_string: str) -> str:
    for part in (
        "XR_ELBOW",
        "XR_FINGER",
        "XR_FOREARM",
        "XR_HAND",
        "XR_HUMERUS",
        "XR_SHOULDER",
        "XR_WRIST",
    ):
        if part in path_string:
            return part.replace("XR_", "").title()

    return "Unknown"


class MURADataset(Dataset):
    def __init__(
        self,
        csv_file: str | Path,
        mura_root: str | Path,
        transform: Callable | None = None,
    ) -> None:
        self.csv_file = Path(csv_file)
        self.mura_root = Path(mura_root)
        self.transform = transform

        if not self.csv_file.exists():
            raise FileNotFoundError(
                f"CSV file not found: {self.csv_file}"
            )

        self.frame = pd.read_csv(
            self.csv_file,
            header=None,
            names=["relative_path"],
        )

        if self.frame.empty:
            raise ValueError(
                f"No image paths found in {self.csv_file}"
            )

        self.frame["relative_path"] = (
            self.frame["relative_path"]
            .astype(str)
            .str.replace("\\", "/", regex=False)
            .str.strip()
        )

        self.frame["label"] = self.frame["relative_path"].map(
            infer_label
        )
        self.frame["body_part"] = self.frame["relative_path"].map(
            infer_body_part
        )

        print(
            f"Loaded {len(self.frame):,} MURA images "
            f"from {self.csv_file.name}."
        )

    def __len__(self) -> int:
        return len(self.frame)

    def _resolve_image_path(self, relative_path: str) -> Path:
        clean_path = relative_path.replace("\\", "/")

        if clean_path.startswith("MURA-v1.1/"):
            clean_path = clean_path[len("MURA-v1.1/"):]

        image_path = self.mura_root / clean_path

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        return image_path

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        row = self.frame.iloc[index]
        image_path = self._resolve_image_path(
            row["relative_path"]
        )

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if self.transform is not None:
                image = self.transform(image)

        label = torch.tensor(
            float(row["label"]),
            dtype=torch.float32,
        )

        return (
            image,
            label,
            row["body_part"],
            str(image_path),
        )
