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

from config import (
    CHEXPERT_ROOT,
    VALID_CSV,
    IMAGE_SIZE,
    NORMAL_LABEL,
    FINDING_LABELS,
    MODEL_PATH,
    THRESHOLDS_PATH,
    METRICS_PATH,
    MIN_SPECIFICITY,
    MIN_POSITIVES_FOR_CALIBRATION,
    NORMAL_GATE_MIN_SPECIFICITY,
)

from dataset import ChestV2Dataset


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 32
NUM_WORKERS = 0

THRESHOLD_GRID = np.linspace(0.05, 0.99, 189)


class ChestV2Model(nn.Module):

    def __init__(self):
        super().__init__()

        backbone = densenet121(weights=None)

        feature_dim = backbone.classifier.in_features

        self.features = backbone.features

        self.normal_head = nn.Linear(
            feature_dim,
            1,
        )

        self.finding_head = nn.Linear(
            feature_dim,
            len(FINDING_LABELS),
        )

    def forward(self, images):

        features = self.features(images)

        features = torch.relu(features)

        pooled = torch.nn.functional.adaptive_avg_pool2d(
            features,
            (1, 1),
        )

        pooled = torch.flatten(pooled, 1)

        normal_logits = self.normal_head(
            pooled
        ).squeeze(1)

        finding_logits = self.finding_head(
            pooled
        )

        return normal_logits, finding_logits


def build_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    expected = checkpoint.get(
        "finding_labels",
        FINDING_LABELS,
    )

    if expected != FINDING_LABELS:
        raise ValueError(
            "Checkpoint labels do not match config.py"
        )

    model = ChestV2Model()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    model.eval()

    return model


def build_loader():

    transform = transforms.Compose(
        [
            transforms.Resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                )
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    dataset = ChestV2Dataset(
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

    return dataset, loader


def collect_predictions(model, loader):

    normal_targets = []
    normal_scores = []
    normal_masks = []

    finding_targets = []
    finding_scores = []
    finding_masks = []

    with torch.inference_mode():

        for (
            images,
            n_target,
            n_mask,
            f_target,
            f_mask,
            _,
        ) in loader:

            images = images.to(DEVICE)

            normal_logits, finding_logits = model(
                images
            )

            normal_prob = torch.sigmoid(
                normal_logits
            )

            finding_prob = torch.sigmoid(
                finding_logits
            )

            normal_targets.append(
                n_target.numpy()
            )

            normal_scores.append(
                normal_prob.cpu().numpy()
            )

            normal_masks.append(
                n_mask.numpy()
            )

            finding_targets.append(
                f_target.numpy()
            )

            finding_scores.append(
                finding_prob.cpu().numpy()
            )

            finding_masks.append(
                f_mask.numpy()
            )

    return (
        np.concatenate(normal_targets),
        np.concatenate(normal_scores),
        np.concatenate(normal_masks),
        np.concatenate(finding_targets),
        np.concatenate(finding_scores),
        np.concatenate(finding_masks),
    )

def metrics_at_threshold(
    y_true,
    y_score,
    threshold,
):
    """
    Compute binary classification metrics at a given threshold.
    """

    y_pred = (
        y_score >= threshold
    ).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    specificity = tn / max(
        tn + fp,
        1,
    )

    sensitivity = tp / max(
        tp + fn,
        1,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def choose_threshold(
    y_true,
    y_score,
    minimum_specificity,
):
    """
    Find the highest-performing threshold while
    maintaining the requested specificity.
    """

    positives = int(y_true.sum())
    negatives = int((1 - y_true).sum())

    if (
        positives < MIN_POSITIVES_FOR_CALIBRATION
        or negatives < 10
    ):
        return (
            0.90,
            "fallback",
        )

    candidates = []

    for threshold in THRESHOLD_GRID:

        result = metrics_at_threshold(
            y_true,
            y_score,
            float(threshold),
        )

        if (
            result["specificity"]
            >= minimum_specificity
        ):
            candidates.append(result)

    if not candidates:
        return (
            0.90,
            "specificity_fallback",
        )

    best = max(
        candidates,
        key=lambda x: (
            x["f1"],
            x["specificity"],
            x["threshold"],
        ),
    )

    return (
        float(best["threshold"]),
        "validation",
    )


def evaluate_normal_gate(
    targets,
    probabilities,
    masks,
):
    """
    Evaluate only the normal-versus-abnormal gate.
    """

    usable = masks.astype(bool)

    y_true = targets[usable].astype(np.int64)
    y_score = probabilities[usable]

    auc = roc_auc_score(
        y_true,
        y_score,
    )

    average_precision = average_precision_score(
        y_true,
        y_score,
    )

    threshold, method = choose_threshold(
        y_true,
        y_score,
        NORMAL_GATE_MIN_SPECIFICITY,
    )

    metrics = metrics_at_threshold(
        y_true,
        y_score,
        threshold,
    )

    threshold_config = {
        "threshold": threshold,
        "negative_max": max(
            0.0,
            threshold - 0.05,
        ),
        "positive_min": min(
            1.0,
            threshold + 0.05,
        ),
        "method": method,
    }

    metric_row = {
        "label": NORMAL_LABEL,
        "auc": float(auc),
        "average_precision": float(
            average_precision
        ),
        **metrics,
    }

    return (
        threshold_config,
        metric_row,
    )


def evaluate_findings(
    targets,
    probabilities,
    masks,
):
    """
    Evaluate each abnormal finding independently.
    """

    threshold_config = {}

    metric_rows = []

    for index, label in enumerate(
        FINDING_LABELS
    ):

        usable = masks[:, index].astype(bool)

        y_true = targets[
            usable,
            index,
        ].astype(np.int64)

        y_score = probabilities[
            usable,
            index,
        ]

        if len(np.unique(y_true)) < 2:
            print(
                f"Skipping {label}"
            )
            continue

        auc = roc_auc_score(
            y_true,
            y_score,
        )

        average_precision = (
            average_precision_score(
                y_true,
                y_score,
            )
        )

        threshold, method = choose_threshold(
            y_true,
            y_score,
            MIN_SPECIFICITY,
        )

        metrics = metrics_at_threshold(
            y_true,
            y_score,
            threshold,
        )

        threshold_config[label] = {
            "threshold": threshold,
            "negative_max": max(
                0.0,
                threshold - 0.05,
            ),
            "positive_min": min(
                1.0,
                threshold + 0.05,
            ),
            "method": method,
        }

        metric_rows.append(
            {
                "label": label,
                "auc": float(auc),
                "average_precision": float(
                    average_precision
                ),
                **metrics,
            }
        )

    return (
        threshold_config,
        metric_rows,
    )

def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find trained Chest V2 model at {MODEL_PATH}. "
            "Run python train.py first."
        )

    print("Using device:", DEVICE)

    dataset, loader = build_loader()

    print(f"Loaded {len(dataset):,} validation images.")

    model = build_model()

    (
        normal_targets,
        normal_probabilities,
        normal_masks,
        finding_targets,
        finding_probabilities,
        finding_masks,
    ) = collect_predictions(
        model,
        loader,
    )

    (
        normal_threshold_config,
        normal_metric_row,
    ) = evaluate_normal_gate(
        normal_targets,
        normal_probabilities,
        normal_masks,
    )

    (
        finding_threshold_config,
        finding_metric_rows,
    ) = evaluate_findings(
        finding_targets,
        finding_probabilities,
        finding_masks,
    )

    threshold_output = {
        "model_version": "chest_v2",
        "architecture": "densenet121_dual_head",
        "normal_gate": {
            NORMAL_LABEL: normal_threshold_config,
        },
        "findings": finding_threshold_config,
    }

    metric_rows = [
        {
            "head": "normal_gate",
            **normal_metric_row,
        }
    ]

    for row in finding_metric_rows:
        metric_rows.append(
            {
                "head": "finding_head",
                **row,
            }
        )

    metrics_frame = pd.DataFrame(metric_rows)

    THRESHOLDS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with THRESHOLDS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            threshold_output,
            file,
            indent=2,
        )

    metrics_frame.to_csv(
        METRICS_PATH,
        index=False,
    )

    print("\nChest V2 validation results")
    print("=" * 90)

    print(
        metrics_frame.to_string(
            index=False,
        )
    )

    print("=" * 90)

    print(
        f"\nSaved thresholds to:\n"
        f"{THRESHOLDS_PATH}"
    )

    print(
        f"\nSaved validation metrics to:\n"
        f"{METRICS_PATH}"
    )

    print(
        "\nEvaluation complete. "
        "Do not connect this model to the backend until "
        "the validation metrics and false-positive behavior "
        "have been reviewed."
    )


if __name__ == "__main__":
    main()