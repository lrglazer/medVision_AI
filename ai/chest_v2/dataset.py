from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from config import (
    CSV_LABELS,
    FINDING_LABELS,
    NORMAL_LABEL,
    UNCERTAINTY_POLICY,
)


class ChestV2Dataset(Dataset):
    def __init__(
        self,
        csv_file,
        chexpert_root,
        transform=None,
        frontal_only=True,
    ):
        self.csv_file = Path(csv_file)
        self.chexpert_root = Path(chexpert_root)
        self.transform = transform

        frame = pd.read_csv(self.csv_file)

        required = {"Path", "Frontal/Lateral", *CSV_LABELS}

        missing = required - set(frame.columns)

        if missing:
            raise KeyError(f"Missing columns: {sorted(missing)}")

        if frontal_only:
            frame = frame[
                frame["Frontal/Lateral"]
                .astype(str)
                .str.lower()
                .eq("frontal")
            ].copy()

        self.frame = frame.reset_index(drop=True)

        print(f"Loaded {len(self.frame):,} frontal studies.")

    def __len__(self):
        return len(self.frame)

    def _resolve_image_path(self, csv_path):
        p = Path(str(csv_path).replace("\\", "/"))

        if p.parts and p.parts[0] == self.chexpert_root.name:
            p = Path(*p.parts[1:])

        image_path = self.chexpert_root / p

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        return image_path

    @staticmethod
    def convert_label(value, policy):

        if pd.isna(value):
            return 0.0, 0.0

        value = float(value)

        if value == 1:
            return 1.0, 1.0

        if value == 0:
            return 0.0, 1.0

        if value == -1:

            if policy == "ones":
                return 1.0, 1.0

            if policy == "zeros":
                return 0.0, 1.0

            if policy == "ignore":
                return 0.0, 0.0

        raise ValueError(value)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]

        image_path = self._resolve_image_path(row["Path"])

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as error:
            raise RuntimeError(
                f"Could not open image: {image_path}"
            ) from error

        if self.transform is not None:
            image = self.transform(image)

        #
        # Finding head
        #

        findings = np.zeros(
            len(FINDING_LABELS),
            dtype=np.float32,
        )

        finding_mask = np.zeros(
            len(FINDING_LABELS),
            dtype=np.float32,
        )

        for i, label in enumerate(FINDING_LABELS):
            target, usable = self.convert_label(
                row[label],
                UNCERTAINTY_POLICY[label],
            )

            findings[i] = target
            finding_mask[i] = usable

        # Derive the normal-gate target.
        #
        # No Finding = 1 means confirmed normal.
        # Any usable positive finding means confirmed abnormal.
        # Otherwise, the normal-gate label is unknown and ignored.
        raw_no_finding = row[NORMAL_LABEL]
        has_positive_finding = bool(
            np.any((findings == 1.0) & (finding_mask == 1.0))
        )

        if not pd.isna(raw_no_finding) and float(raw_no_finding) == 1.0:
            normal_target = 1.0
            normal_mask = 1.0
        elif has_positive_finding:
            normal_target = 0.0
            normal_mask = 1.0
        elif not pd.isna(raw_no_finding) and float(raw_no_finding) == 0.0:
            normal_target = 0.0
            normal_mask = 1.0
        else:
            normal_target = 0.0
            normal_mask = 0.0

        return (
            image,
            torch.tensor(normal_target, dtype=torch.float32),
            torch.tensor(normal_mask, dtype=torch.float32),
            torch.from_numpy(findings),
            torch.from_numpy(finding_mask),
            str(image_path),
        )