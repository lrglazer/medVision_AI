from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ai.shared.study_validator import (
    BinaryImageDataset,
    build_validator_model,
    create_balanced_records,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--positive-dir", action="append", required=True)
    parser.add_argument("--negative-dir", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--target-specificity", type=float, default=0.98)
    parser.add_argument("--max-images-per-class", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_images(raw_paths: list[str], max_images: int) -> list[Path]:
    images: list[Path] = []

    for raw_path in raw_paths:
        folder = Path(raw_path)

        if not folder.exists():
            print(f"Warning: folder not found: {folder}", flush=True)
            continue

        for path in folder.rglob("*"):
            if not path.is_file():
                continue
            if path.name.startswith("._"):
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            images.append(path)

            if len(images) >= max_images:
                return images

    return images


def choose_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    target_specificity: float,
) -> float:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)
    specificity = 1.0 - false_positive_rate
    valid_indices = np.where(specificity >= target_specificity)[0]

    if len(valid_indices) == 0:
        return 0.5

    best_index = valid_indices[np.argmax(true_positive_rate[valid_indices])]
    threshold = float(thresholds[best_index])

    if not np.isfinite(threshold):
        return 0.5

    return max(0.05, min(0.95, threshold))


def evaluate(model, loader, device):
    model.eval()
    labels: list[float] = []
    scores: list[float] = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images).squeeze(1)
            probabilities = torch.sigmoid(logits).cpu().numpy()

            labels.extend(targets.numpy().tolist())
            scores.extend(probabilities.tolist())

    return np.asarray(labels), np.asarray(scores)


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Scanning positive images...", flush=True)
    positives = collect_images(
        args.positive_dir,
        max_images=args.max_images_per_class,
    )

    print("Scanning negative images...", flush=True)
    negatives = collect_images(
        args.negative_dir,
        max_images=args.max_images_per_class,
    )

    random.shuffle(positives)
    random.shuffle(negatives)

    print(f"Positive images found: {len(positives):,}", flush=True)
    print(f"Negative images found: {len(negatives):,}", flush=True)

    if not positives:
        raise RuntimeError("No positive images were found.")
    if not negatives:
        raise RuntimeError("No negative images were found.")

    records = create_balanced_records(
        positives,
        negatives,
        seed=args.seed,
    )

    split_index = int(len(records) * 0.80)
    train_records = records[:split_index]
    validation_records = records[split_index:]

    print(
        f"Training records: {len(train_records):,} | "
        f"Validation records: {len(validation_records):,}",
        flush=True,
    )

    train_dataset = BinaryImageDataset(
        train_records,
        image_size=args.image_size,
        training=True,
    )
    validation_dataset = BinaryImageDataset(
        validation_records,
        image_size=args.image_size,
        training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}", flush=True)

    model = build_validator_model(pretrained=True).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    loss_function = nn.BCEWithLogitsLoss()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_auc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        print(f"Starting epoch {epoch}/{args.epochs}...", flush=True)

        for batch_index, (images, targets) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images).squeeze(1)
            loss = loss_function(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            if batch_index % 25 == 0 or batch_index == len(train_loader):
                print(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"batch {batch_index}/{len(train_loader)}",
                    flush=True,
                )

        labels, scores = evaluate(model, validation_loader, device)
        auc = roc_auc_score(labels, scores)
        threshold = choose_threshold(
            labels,
            scores,
            target_specificity=args.target_specificity,
        )
        predictions = (scores >= threshold).astype(np.int64)

        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0)
        sensitivity = recall_score(labels, predictions, zero_division=0)

        negative_mask = labels == 0
        specificity = float(np.mean(predictions[negative_mask] == 0))

        average_loss = running_loss / max(1, len(train_dataset))

        print(
            f"Epoch {epoch:02d} | "
            f"loss={average_loss:.4f} | "
            f"auc={auc:.4f} | "
            f"accuracy={accuracy:.4f} | "
            f"sensitivity={sensitivity:.4f} | "
            f"specificity={specificity:.4f} | "
            f"threshold={threshold:.4f}",
            flush=True,
        )

        if auc > best_auc:
            best_auc = auc

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "image_size": args.image_size,
                "threshold": float(threshold),
                "validation_auc": float(auc),
                "validation_accuracy": float(accuracy),
                "validation_precision": float(precision),
                "validation_sensitivity": float(sensitivity),
                "validation_specificity": float(specificity),
                "positive_class": args.name,
                "negative_class": f"not_{args.name}",
            }
            torch.save(checkpoint, output_path)

            output_path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        key: value
                        for key, value in checkpoint.items()
                        if key != "model_state_dict"
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            print(f"Saved new best checkpoint: {output_path}", flush=True)

    print("Training complete.", flush=True)


if __name__ == "__main__":
    main()