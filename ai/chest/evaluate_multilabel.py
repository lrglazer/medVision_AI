from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import densenet121

from chexpert_config import (
    CHEXPERT_ROOT,
    IMAGE_SIZE,
    LABELS,
    MODEL_PATH,
    UNCERTAINTY_POLICY,
    VALID_CSV,
)
from dataset_multilabel import CheXpertMultilabelDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_WORKERS = 0
THRESHOLDS_PATH = Path("models/chexpert_thresholds.json")
METRICS_PATH = Path("models/chexpert_validation_metrics.csv")

# Avoid thresholds that produce extreme false-positive rates.
MIN_SPECIFICITY = 0.85
MIN_POSITIVES_FOR_CALIBRATION = 10
THRESHOLD_GRID = np.linspace(0.05, 0.99, 189)


def build_model() -> nn.Module:
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )
    labels = checkpoint.get("labels", LABELS)
    if labels != LABELS:
        raise ValueError(
            "Checkpoint label order does not match chexpert_config.py."
        )

    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(LABELS))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    return model


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []
    masks_all: list[np.ndarray] = []

    with torch.inference_mode():
        for images, targets, mask, _paths in loader:
            images = images.to(DEVICE, non_blocking=True)
            logits = model(images)
            probabilities = torch.sigmoid(logits)

            targets_all.append(targets.numpy())
            probabilities_all.append(probabilities.cpu().numpy())
            masks_all.append(mask.numpy())

    return (
        np.concatenate(targets_all),
        np.concatenate(probabilities_all),
        np.concatenate(masks_all),
    )


def metrics_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    y_pred = (y_score >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    specificity = tn / max(tn + fp, 1)
    sensitivity = tp / max(tp + fn, 1)

    return {
        "threshold": float(threshold),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
    }


def choose_threshold(
    label: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[float, str]:
    positive_count = int(y_true.sum())
    negative_count = int((1 - y_true).sum())

    if positive_count < MIN_POSITIVES_FOR_CALIBRATION or negative_count < 10:
        # Rare validation classes cannot support reliable threshold fitting.
        # A conservative threshold reduces wild false positives.
        return 0.90, "conservative_fallback"

    candidates: list[dict[str, float | int]] = []
    for threshold in THRESHOLD_GRID:
        result = metrics_at_threshold(y_true, y_score, float(threshold))
        if float(result["specificity"]) >= MIN_SPECIFICITY:
            candidates.append(result)

    if not candidates:
        return 0.90, "specificity_fallback"

    # Prefer F1, then specificity, then the higher threshold.
    best = max(
        candidates,
        key=lambda item: (
            float(item["f1"]),
            float(item["specificity"]),
            float(item["threshold"]),
        ),
    )

    threshold = float(best["threshold"])

    # "No Finding" is a separate positive label and behaves differently.
    # Keep it conservative so it does not claim normality too easily.
    if label == "No Finding":
        threshold = max(threshold, 0.80)

    return threshold, "validation_f1_with_specificity_floor"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {MODEL_PATH}. Train the model first."
        )

    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    dataset = CheXpertMultilabelDataset(
        csv_file=VALID_CSV,
        chexpert_root=CHEXPERT_ROOT,
        transform=transform,
        frontal_only=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Loaded {len(dataset)} validation images.")
    print("Using device:", DEVICE)

    model = build_model()
    targets, probabilities, masks = collect_predictions(model, loader)

    threshold_config: dict[str, dict[str, float | str | int]] = {}
    metric_rows: list[dict[str, float | str | int]] = []

    for index, label in enumerate(LABELS):
        usable = masks[:, index].astype(bool)
        y_true = targets[usable, index].astype(np.int64)
        y_score = probabilities[usable, index]

        positive_count = int(y_true.sum())
        negative_count = int((1 - y_true).sum())

        if len(np.unique(y_true)) < 2:
            print(f"Skipping {label}: validation set has one class only.")
            continue

        auc = float(roc_auc_score(y_true, y_score))
        average_precision = float(
            average_precision_score(y_true, y_score)
        )
        threshold, method = choose_threshold(label, y_true, y_score)
        result = metrics_at_threshold(y_true, y_score, threshold)

        # Preserve the three-band interface expected by backend/main.py.
        negative_max = max(0.0, threshold - 0.05)
        positive_min = min(1.0, threshold + 0.05)

        threshold_config[label] = {
            "threshold": threshold,
            "negative_max": negative_max,
            "positive_min": positive_min,
            "method": method,
            "validation_positives": positive_count,
            "validation_negatives": negative_count,
        }

        metric_rows.append(
            {
                "label": label,
                "auc": auc,
                "average_precision": average_precision,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "calibration_method": method,
                **result,
            }
        )

    metrics_frame = pd.DataFrame(metric_rows)
    print(metrics_frame.to_string(index=False))

    THRESHOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with THRESHOLDS_PATH.open("w", encoding="utf-8") as file:
        json.dump(threshold_config, file, indent=2)

    metrics_frame.to_csv(METRICS_PATH, index=False)

    print(f"\nSaved thresholds to {THRESHOLDS_PATH}")
    print(f"Saved validation metrics to {METRICS_PATH}")
    print("\nRestart the backend after these files are generated.")


if __name__ == "__main__":
    main()
