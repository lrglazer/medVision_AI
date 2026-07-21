from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import densenet121

from bone_config import (
    BEST_MODEL_PATH,
    BATCH_SIZE,
    IMAGE_SIZE,
    METRICS_PATH,
    MURA_ROOT,
    NUM_WORKERS,
    THRESHOLD_PATH,
    VALID_IMAGE_PATHS,
)
from dataset_mura import MURADataset


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def build_model() -> nn.Module:
    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model = densenet121(weights=None)
    model.classifier = nn.Linear(
        model.classifier.in_features,
        1,
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model = model.to(DEVICE)
    model.eval()

    return model


def choose_threshold(
    targets: np.ndarray,
    scores: np.ndarray,
) -> float:
    false_positive_rate, true_positive_rate, thresholds = (
        roc_curve(targets, scores)
    )

    youden_index = true_positive_rate - false_positive_rate
    best_index = int(np.argmax(youden_index))

    threshold = float(thresholds[best_index])

    return min(max(threshold, 0.01), 0.99)


def main() -> None:
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

    dataset = MURADataset(
        csv_file=VALID_IMAGE_PATHS,
        mura_root=MURA_ROOT,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model()

    targets_list: list[np.ndarray] = []
    scores_list: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels, _body_parts, _paths in loader:
            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            logits = model(images)
            scores = torch.sigmoid(logits)

            targets_list.append(
                labels.numpy()
            )
            scores_list.append(
                scores.cpu().numpy().ravel()
            )

    targets = np.concatenate(targets_list)
    scores = np.concatenate(scores_list)

    threshold = choose_threshold(
        targets,
        scores,
    )
    predictions = (
        scores >= threshold
    ).astype(int)

    auc = float(
        roc_auc_score(targets, scores)
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            targets,
            predictions,
            average="binary",
            zero_division=0,
        )
    )

    true_negative, false_positive, false_negative, true_positive = (
        confusion_matrix(
            targets,
            predictions,
            labels=[0, 1],
        ).ravel()
    )

    specificity = (
        true_negative
        / max(true_negative + false_positive, 1)
    )

    metrics = {
        "auc": auc,
        "threshold": threshold,
        "precision": float(precision),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "true_positive": int(true_positive),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
    }

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    THRESHOLD_PATH.write_text(
        json.dumps(
            {
                "threshold": threshold,
                "negative_max": max(
                    0.0,
                    threshold - 0.10,
                ),
                "positive_min": min(
                    1.0,
                    threshold + 0.10,
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2))
    print()
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Saved threshold to {THRESHOLD_PATH}")


if __name__ == "__main__":
    main()
