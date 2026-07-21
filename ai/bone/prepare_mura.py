from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from bone_config import (
    MURA_ROOT,
    TRAIN_IMAGE_PATHS,
    VALID_IMAGE_PATHS,
)
from dataset_mura import infer_body_part, infer_label


def inspect_split(csv_path: Path, split_name: str) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing {split_name} CSV: {csv_path}"
        )

    frame = pd.read_csv(
        csv_path,
        header=None,
        names=["relative_path"],
    )

    frame["relative_path"] = (
        frame["relative_path"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
        .str.strip()
    )

    labels = frame["relative_path"].map(infer_label)
    body_parts = frame["relative_path"].map(infer_body_part)

    missing_paths: list[str] = []

    for path_string in frame["relative_path"]:
        clean_path = path_string

        if clean_path.startswith("MURA-v1.1/"):
            clean_path = clean_path[len("MURA-v1.1/"):]

        image_path = MURA_ROOT / clean_path

        if not image_path.exists():
            missing_paths.append(str(image_path))

            if len(missing_paths) >= 10:
                break

    print()
    print(split_name.upper())
    print("-" * 56)
    print(f"Images: {len(frame):,}")
    print(f"Positive: {int(labels.sum()):,}")
    print(f"Negative: {int((labels == 0).sum()):,}")
    print("Body parts:")

    for body_part, count in Counter(body_parts).most_common():
        print(f"  {body_part:12s} {count:,}")

    if missing_paths:
        print()
        print("Example missing files:")

        for path in missing_paths:
            print(f"  {path}")

        raise FileNotFoundError(
            "Some CSV paths do not match the extracted dataset."
        )

    print("All sampled image paths exist.")


def main() -> None:
    print("MURA root:", MURA_ROOT)

    if not MURA_ROOT.exists():
        raise FileNotFoundError(
            f"MURA root does not exist: {MURA_ROOT}"
        )

    inspect_split(TRAIN_IMAGE_PATHS, "train")
    inspect_split(VALID_IMAGE_PATHS, "valid")

    print()
    print("MURA dataset validation complete.")


if __name__ == "__main__":
    main()
