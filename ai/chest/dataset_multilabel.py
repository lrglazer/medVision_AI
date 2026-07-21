from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from chexpert_config import LABELS, UNCERTAINTY_POLICY


class CheXpertMultilabelDataset(Dataset):
    def __init__(self, csv_file, chexpert_root, transform=None, frontal_only=True):
        self.csv_file = Path(csv_file)
        self.chexpert_root = Path(chexpert_root)
        self.transform = transform

        if not self.csv_file.exists():
            raise FileNotFoundError(f"Could not find CSV file: {self.csv_file}")

        frame = pd.read_csv(self.csv_file)
        required = {"Path", *LABELS}
        missing = required - set(frame.columns)

        if missing:
            raise KeyError(f"CSV is missing columns: {sorted(missing)}")

        if frontal_only:
            frame = frame[
                frame["Frontal/Lateral"].astype(str).str.lower().eq("frontal")
            ].copy()

        self.frame = frame.reset_index(drop=True)

        if self.frame.empty:
            raise RuntimeError("No usable CheXpert rows were found.")

        print(f"Loaded {len(self.frame):,} images from {self.csv_file.name}.")

    def __len__(self):
        return len(self.frame)

    def _resolve_image_path(self, csv_path):
        normalized = Path(str(csv_path).replace("\\", "/"))
        if normalized.parts and normalized.parts[0] == self.chexpert_root.name:
            normalized = Path(*normalized.parts[1:])
        image_path = self.chexpert_root / normalized

        if not image_path.exists():
            raise FileNotFoundError(f"Image does not exist: {image_path}")

        return image_path

    @staticmethod
    def convert_label(value, policy):
        if pd.isna(value):
            return 0.0, 0.0

        numeric = float(value)

        if numeric == 1.0:
            return 1.0, 1.0
        if numeric == 0.0:
            return 0.0, 1.0
        if numeric == -1.0:
            if policy == "ones":
                return 1.0, 1.0
            if policy == "zeros":
                return 0.0, 1.0
            if policy == "ignore":
                return 0.0, 0.0

        raise ValueError(f"Unexpected label value: {value}")

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image_path = self._resolve_image_path(row["Path"])

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as error:
            raise RuntimeError(f"Could not open image: {image_path}") from error

        targets = np.zeros(len(LABELS), dtype=np.float32)
        mask = np.zeros(len(LABELS), dtype=np.float32)

        for label_index, label_name in enumerate(LABELS):
            target, usable = self.convert_label(
                row[label_name],
                UNCERTAINTY_POLICY[label_name],
            )
            targets[label_index] = target
            mask[label_index] = usable

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.from_numpy(targets), torch.from_numpy(mask), str(image_path)
