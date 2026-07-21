from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import DenseNet121_Weights, densenet121

from chexpert_config import (
    CHEXPERT_ROOT,
    IMAGE_SIZE,
    LABELS,
    MODEL_PATH,
    TRAIN_CSV,
    UNCERTAINTY_POLICY,
    VALID_CSV,
)
from dataset_multilabel import CheXpertMultilabelDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("models/checkpoints")
LAST_CHECKPOINT_PATH = CHECKPOINT_DIR / "last.pt"

DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 24
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 4
NUM_WORKERS = 0
SEED = 42


class MaskedBCEWithLogitsLoss(nn.Module):
    """Binary cross-entropy that ignores unusable CheXpert labels."""

    def __init__(self, pos_weight: torch.Tensor) -> None:
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(
            reduction="none",
            pos_weight=pos_weight,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        elementwise = self.loss(logits, targets)
        masked = elementwise * mask
        return masked.sum() / mask.sum().clamp_min(1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the MedVision DenseNet-121 CheXpert model."
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    return parser.parse_args()


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms():
    # Mild augmentations only. Large rotations/crops can create fake pathology.
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=3),
            transforms.ColorJitter(brightness=0.08, contrast=0.08),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    validation_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return train_transform, validation_transform


def build_model() -> nn.Module:
    model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, len(LABELS))
    return model.to(DEVICE)


def compute_pos_weight(
    dataset: CheXpertMultilabelDataset,
) -> torch.Tensor:
    positives = np.zeros(len(LABELS), dtype=np.float64)
    negatives = np.zeros(len(LABELS), dtype=np.float64)

    for _, row in dataset.frame.iterrows():
        for index, label in enumerate(LABELS):
            target, usable = dataset.convert_label(
                row[label],
                UNCERTAINTY_POLICY[label],
            )
            if not usable:
                continue
            if target == 1:
                positives[index] += 1
            else:
                negatives[index] += 1

    raw_ratio = negatives / np.clip(positives, 1.0, None)

    # Square-root compression prevents extremely rare labels from dominating
    # the loss and causing the widespread false positives seen previously.
    weights = np.sqrt(raw_ratio)
    weights = np.clip(weights, 1.0, 8.0)

    print("\nPositive-class weights")
    print("-" * 52)
    for label, positive_count, weight in zip(LABELS, positives, weights):
        print(f"{label:28s} positives={positive_count:8.0f} weight={weight:6.3f}")

    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def validation_pass(
    model: nn.Module,
    loader: DataLoader,
    loss_function: MaskedBCEWithLogitsLoss,
) -> tuple[float, float, dict[str, float]]:
    model.eval()
    losses: list[float] = []
    targets_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []
    masks_all: list[np.ndarray] = []

    with torch.inference_mode():
        for images, targets, mask, _paths in loader:
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)

            with torch.amp.autocast(
                "cuda",
                enabled=torch.cuda.is_available(),
            ):
                logits = model(images)
                loss = loss_function(logits, targets, mask)

            losses.append(float(loss.item()))
            targets_all.append(targets.cpu().numpy())
            probabilities_all.append(torch.sigmoid(logits).cpu().numpy())
            masks_all.append(mask.cpu().numpy())

    targets_array = np.concatenate(targets_all)
    probabilities_array = np.concatenate(probabilities_all)
    masks_array = np.concatenate(masks_all)

    per_label_auc: dict[str, float] = {}
    for index, label in enumerate(LABELS):
        usable = masks_array[:, index].astype(bool)
        y_true = targets_array[usable, index]
        y_score = probabilities_array[usable, index]
        if len(np.unique(y_true)) >= 2:
            per_label_auc[label] = float(roc_auc_score(y_true, y_score))

    macro_auc = float(np.mean(list(per_label_auc.values()))) if per_label_auc else 0.0
    return float(np.mean(losses)), macro_auc, per_label_auc


def save_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    completed_epoch: int,
    best_macro_auc: float,
    epochs_without_improvement: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "completed_epoch": completed_epoch,
            "best_macro_auc": best_macro_auc,
            "epochs_without_improvement": epochs_without_improvement,
            "labels": LABELS,
            "image_size": IMAGE_SIZE,
            "architecture": "densenet121",
        },
        path,
    )


def save_best_model(model: nn.Module, macro_auc: float) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "labels": LABELS,
            "image_size": IMAGE_SIZE,
            "architecture": "densenet121",
            "validation_macro_auc": macro_auc,
        },
        MODEL_PATH,
    )


def main() -> None:
    args = parse_args()
    set_seed()

    print("Using device:", DEVICE)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("CPU training will be slow. You can stop safely and resume later.")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_transform, validation_transform = build_transforms()

    train_dataset = CheXpertMultilabelDataset(
        csv_file=TRAIN_CSV,
        chexpert_root=CHEXPERT_ROOT,
        transform=train_transform,
        frontal_only=True,
    )
    validation_dataset = CheXpertMultilabelDataset(
        csv_file=VALID_CSV,
        chexpert_root=CHEXPERT_ROOT,
        transform=validation_transform,
        frontal_only=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model()
    pos_weight = compute_pos_weight(train_dataset)
    loss_function = MaskedBCEWithLogitsLoss(pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=DEFAULT_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=torch.cuda.is_available(),
    )

    start_epoch = 1
    best_macro_auc = -1.0
    epochs_without_improvement = 0

    if args.resume:
        if not LAST_CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {LAST_CHECKPOINT_PATH}"
            )

        checkpoint = torch.load(
            LAST_CHECKPOINT_PATH,
            map_location=DEVICE,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        start_epoch = int(checkpoint["completed_epoch"]) + 1
        best_macro_auc = float(checkpoint.get("best_macro_auc", -1.0))
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        print(f"Resuming at epoch {start_epoch}.")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        batch_losses: list[float] = []

        for batch_number, (images, targets, mask, _paths) in enumerate(
            train_loader,
            start=1,
        ):
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                "cuda",
                enabled=torch.cuda.is_available(),
            ):
                logits = model(images)
                loss = loss_function(logits, targets, mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            batch_losses.append(float(loss.item()))

            if batch_number % 100 == 0 or batch_number == len(train_loader):
                print(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"Batch {batch_number}/{len(train_loader)} | "
                    f"Loss {loss.item():.4f}",
                    flush=True,
                )

        validation_loss, validation_macro_auc, per_label_auc = validation_pass(
            model,
            validation_loader,
            loss_function,
        )
        scheduler.step(validation_macro_auc)

        training_loss = float(np.mean(batch_losses))
        learning_rate = optimizer.param_groups[0]["lr"]

        print("\n" + "=" * 62)
        print(f"Epoch {epoch} complete")
        print(f"Training loss:        {training_loss:.4f}")
        print(f"Validation loss:      {validation_loss:.4f}")
        print(f"Validation macro AUC: {validation_macro_auc:.4f}")
        print(f"Learning rate:        {learning_rate:.2e}")
        print("-" * 62)
        for label, auc in per_label_auc.items():
            print(f"{label:28s} AUC={auc:.4f}")
        print("=" * 62 + "\n")

        improved = validation_macro_auc > best_macro_auc + 1e-4
        if improved:
            best_macro_auc = validation_macro_auc
            epochs_without_improvement = 0
            save_best_model(model, best_macro_auc)
            print(f"Saved new best model to {MODEL_PATH}")
        else:
            epochs_without_improvement += 1
            print(
                f"No improvement for {epochs_without_improvement} epoch(s)."
            )

        save_checkpoint(
            path=CHECKPOINT_DIR / f"epoch_{epoch}.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            completed_epoch=epoch,
            best_macro_auc=best_macro_auc,
            epochs_without_improvement=epochs_without_improvement,
        )
        save_checkpoint(
            path=LAST_CHECKPOINT_PATH,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            completed_epoch=epoch,
            best_macro_auc=best_macro_auc,
            epochs_without_improvement=epochs_without_improvement,
        )

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping after {args.patience} epochs without improvement."
            )
            break

    print(f"Training finished. Best validation macro AUC: {best_macro_auc:.4f}")
    print("Next run: python evaluate_multilabel.py")


if __name__ == "__main__":
    main()
