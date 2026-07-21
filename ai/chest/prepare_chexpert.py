from pathlib import Path

import pandas as pd

from chexpert_config import CHEXPERT_ROOT, LABELS, TRAIN_CSV, VALID_CSV


def validate_csv(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find {csv_path}. Download and extract CheXpert first."
        )

    frame = pd.read_csv(csv_path)
    required = {"Path", "Frontal/Lateral", *LABELS}
    missing = required - set(frame.columns)

    if missing:
        raise KeyError(f"{csv_path.name} is missing: {sorted(missing)}")

    return frame


def summarize(name, frame):
    frontal = frame[
        frame["Frontal/Lateral"].astype(str).str.lower().eq("frontal")
    ].copy()

    print(f"\n{name}")
    print(f"All rows: {len(frame):,}")
    print(f"Frontal rows: {len(frontal):,}")

    for label in LABELS:
        positives = int((frontal[label] == 1).sum())
        uncertain = int((frontal[label] == -1).sum())
        print(f"{label:28s} positive={positives:6d} uncertain={uncertain:6d}")


def check_paths(frame, sample_size=500):
    for csv_path in frame.head(sample_size)["Path"]:
        normalized = Path(str(csv_path).replace("\\", "/"))
        if normalized.parts and normalized.parts[0] == CHEXPERT_ROOT.name:
            normalized = Path(*normalized.parts[1:])
        image_path = CHEXPERT_ROOT / normalized
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")


def main():
    train_frame = validate_csv(TRAIN_CSV)
    valid_frame = validate_csv(VALID_CSV)

    summarize("TRAINING", train_frame)
    summarize("VALIDATION", valid_frame)

    check_paths(train_frame)
    check_paths(valid_frame)

    print("\nCheXpert files and sample image paths look valid.")


if __name__ == "__main__":
    main()
