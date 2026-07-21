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

from config import (
    CHECKPOINT_DIR,
    CHEXPERT_ROOT,
    FINDING_LABELS,
    IMAGE_SIZE,
    LAST_CHECKPOINT_PATH,
    MODEL_PATH,
    NORMAL_LABEL,
    TRAIN_CSV,
    UNCERTAINTY_POLICY,
    VALID_CSV,
)
from dataset import ChestV2Dataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 24
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 4

NUM_WORKERS = 0
SEED = 42


class ChestV2Model(nn.Module):
    """
    DenseNet-121 with two prediction heads:

    1. Normal head:
       Predicts the probability that the study has "No Finding."

    2. Finding head:
       Predicts the eight supported abnormal findings.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        backbone = densenet121(weights=weights)

        feature_count = backbone.classifier.in_features

        # Keep the DenseNet feature extractor.
        self.features = backbone.features

        # Replace the original classifier with two separate heads.
        self.normal_head = nn.Linear(feature_count, 1)
        self.finding_head = nn.Linear(
            feature_count,
            len(FINDING_LABELS),
        )

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        features = self.features(images)

        features = torch.relu(features)

        pooled = torch.nn.functional.adaptive_avg_pool2d(
            features,
            output_size=(1, 1),
        )

        pooled = torch.flatten(pooled, 1)

        normal_logits = self.normal_head(pooled).squeeze(1)
        finding_logits = self.finding_head(pooled)

        return normal_logits, finding_logits


class MaskedBinaryLoss(nn.Module):
    """
    Binary cross-entropy that ignores targets whose mask value is zero.
    """

    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
    ):
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

        elementwise_loss = self.loss(logits, targets)
        masked_loss = elementwise_loss * mask

        return masked_loss.sum() / mask.sum().clamp_min(1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the MedVision Chest V2 model."
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from models_v2/checkpoints/last.pt.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
    )

    return parser.parse_args()


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms():
    """
    Use mild augmentation.

    Large rotations, crops, or perspective changes can create artificial
    radiographic findings, so the transformations remain conservative.
    """

    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=3),
            transforms.ColorJitter(
                brightness=0.08,
                contrast=0.08,
            ),
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


def compute_class_weights(
    dataset: ChestV2Dataset,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate positive-class weights from the usable CheXpert labels.

    Square-root compression prevents rare classes from dominating the loss
    and causing excessive positive predictions.
    """

    normal_positives = 0.0
    normal_negatives = 0.0

    finding_positives = np.zeros(
        len(FINDING_LABELS),
        dtype=np.float64,
    )

    finding_negatives = np.zeros(
        len(FINDING_LABELS),
        dtype=np.float64,
    )

    for _, row in dataset.frame.iterrows():

        normal_target, normal_usable = dataset.convert_label(
            row[NORMAL_LABEL],
            UNCERTAINTY_POLICY[NORMAL_LABEL],
        )

        if normal_usable:
            if normal_target == 1:
                normal_positives += 1
            else:
                normal_negatives += 1

        for index, label in enumerate(FINDING_LABELS):

            target, usable = dataset.convert_label(
                row[label],
                UNCERTAINTY_POLICY[label],
            )

            if not usable:
                continue

            if target == 1:
                finding_positives[index] += 1
            else:
                finding_negatives[index] += 1

    normal_ratio = normal_negatives / max(normal_positives, 1.0)
    normal_weight = np.sqrt(normal_ratio)
    normal_weight = float(np.clip(normal_weight, 1.0, 8.0))

    finding_ratios = finding_negatives / np.clip(
        finding_positives,
        1.0,
        None,
    )

    finding_weights = np.sqrt(finding_ratios)
    finding_weights = np.clip(finding_weights, 1.0, 8.0)

    print("\nNormal-gate class weight")
    print("-" * 60)
    print(
        f"{NORMAL_LABEL:28s} "
        f"positives={normal_positives:8.0f} "
        f"negatives={normal_negatives:8.0f} "
        f"weight={normal_weight:6.3f}"
    )

    print("\nFinding-head class weights")
    print("-" * 60)

    for label, positive_count, negative_count, weight in zip(
        FINDING_LABELS,
        finding_positives,
        finding_negatives,
        finding_weights,
    ):
        print(
            f"{label:28s} "
            f"positives={positive_count:8.0f} "
            f"negatives={negative_count:8.0f} "
            f"weight={weight:6.3f}"
        )

    normal_tensor = torch.tensor(
        [normal_weight],
        dtype=torch.float32,
        device=DEVICE,
    )

    finding_tensor = torch.tensor(
        finding_weights,
        dtype=torch.float32,
        device=DEVICE,
    )

    return normal_tensor, finding_tensor


def calculate_auc(
    targets: np.ndarray,
    probabilities: np.ndarray,
    mask: np.ndarray,
) -> float | None:

    usable = mask.astype(bool)

    y_true = targets[usable]
    y_score = probabilities[usable]

    if len(y_true) == 0:
        return None

    if len(np.unique(y_true)) < 2:
        return None

    return float(roc_auc_score(y_true, y_score))


def validation_pass(
    model: ChestV2Model,
    loader: DataLoader,
    normal_loss_function: MaskedBinaryLoss,
    finding_loss_function: MaskedBinaryLoss,
) -> tuple[
    float,
    float | None,
    float,
    float,
    dict[str, float],
]:
    model.eval()

    losses: list[float] = []

    normal_targets_all: list[np.ndarray] = []
    normal_probabilities_all: list[np.ndarray] = []
    normal_masks_all: list[np.ndarray] = []

    finding_targets_all: list[np.ndarray] = []
    finding_probabilities_all: list[np.ndarray] = []
    finding_masks_all: list[np.ndarray] = []

    with torch.inference_mode():

        for (
            images,
            normal_targets,
            normal_masks,
            finding_targets,
            finding_masks,
            _paths,
        ) in loader:

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            normal_targets = normal_targets.to(
                DEVICE,
                non_blocking=True,
            )

            normal_masks = normal_masks.to(
                DEVICE,
                non_blocking=True,
            )

            finding_targets = finding_targets.to(
                DEVICE,
                non_blocking=True,
            )

            finding_masks = finding_masks.to(
                DEVICE,
                non_blocking=True,
            )

            with torch.amp.autocast(
                "cuda",
                enabled=torch.cuda.is_available(),
            ):
                normal_logits, finding_logits = model(images)

                normal_loss = normal_loss_function(
                    normal_logits,
                    normal_targets,
                    normal_masks,
                )

                # Confirmed normal studies should not train the finding head
                # to produce abnormal findings.
                effective_finding_mask = finding_masks

                finding_loss = finding_loss_function(
                    finding_logits,
                    finding_targets,
                    effective_finding_mask,
                )

                total_loss = normal_loss + finding_loss

            losses.append(float(total_loss.item()))

            normal_targets_all.append(
                normal_targets.cpu().numpy()
            )

            normal_probabilities_all.append(
                torch.sigmoid(normal_logits).cpu().numpy()
            )

            normal_masks_all.append(
                normal_masks.cpu().numpy()
            )

            finding_targets_all.append(
                finding_targets.cpu().numpy()
            )

            finding_probabilities_all.append(
                torch.sigmoid(finding_logits).cpu().numpy()
            )

            finding_masks_all.append(
                effective_finding_mask.cpu().numpy()
            )

    normal_targets_array = np.concatenate(normal_targets_all)
    normal_probabilities_array = np.concatenate(
        normal_probabilities_all
    )
    normal_masks_array = np.concatenate(normal_masks_all)

    finding_targets_array = np.concatenate(finding_targets_all)
    finding_probabilities_array = np.concatenate(
        finding_probabilities_all
    )
    finding_masks_array = np.concatenate(finding_masks_all)

    normal_auc = calculate_auc(
        normal_targets_array,
        normal_probabilities_array,
        normal_masks_array,
    )

    per_finding_auc: dict[str, float] = {}

    for index, label in enumerate(FINDING_LABELS):

        auc = calculate_auc(
            finding_targets_array[:, index],
            finding_probabilities_array[:, index],
            finding_masks_array[:, index],
        )

        if auc is not None:
            per_finding_auc[label] = auc

    if per_finding_auc:
        findings_macro_auc = float(
            np.mean(list(per_finding_auc.values()))
        )
    else:
        findings_macro_auc = 0.0

    available_auc_values = list(per_finding_auc.values())

    if normal_auc is not None:
        available_auc_values.append(normal_auc)

    if available_auc_values:
        combined_auc = float(np.mean(available_auc_values))
    else:
        combined_auc = 0.0

    validation_loss = float(np.mean(losses))

    return (
        validation_loss,
        normal_auc,
        findings_macro_auc,
        combined_auc,
        per_finding_auc,
    )


def save_checkpoint(
    *,
    path: Path,
    model: ChestV2Model,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    completed_epoch: int,
    best_combined_auc: float,
    epochs_without_improvement: int,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "completed_epoch": completed_epoch,
            "best_combined_auc": best_combined_auc,
            "epochs_without_improvement": (
                epochs_without_improvement
            ),
            "normal_label": NORMAL_LABEL,
            "finding_labels": FINDING_LABELS,
            "image_size": IMAGE_SIZE,
            "architecture": "densenet121_dual_head",
        },
        path,
    )


def save_best_model(
    model: ChestV2Model,
    combined_auc: float,
    normal_auc: float | None,
    findings_macro_auc: float,
) -> None:

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "normal_label": NORMAL_LABEL,
            "finding_labels": FINDING_LABELS,
            "image_size": IMAGE_SIZE,
            "architecture": "densenet121_dual_head",
            "validation_combined_auc": combined_auc,
            "validation_normal_auc": normal_auc,
            "validation_findings_macro_auc": findings_macro_auc,
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
        print(
            "CPU training will be slow. "
            "You can stop safely and resume later."
        )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_transform, validation_transform = build_transforms()

    train_dataset = ChestV2Dataset(
        csv_file=TRAIN_CSV,
        chexpert_root=CHEXPERT_ROOT,
        transform=train_transform,
        frontal_only=True,
    )

    validation_dataset = ChestV2Dataset(
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

    model = ChestV2Model(pretrained=True).to(DEVICE)

    normal_pos_weight, finding_pos_weights = (
        compute_class_weights(train_dataset)
    )

    normal_loss_function = MaskedBinaryLoss(
        pos_weight=normal_pos_weight,
    )

    finding_loss_function = MaskedBinaryLoss(
        pos_weight=finding_pos_weights,
    )

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
    best_combined_auc = -1.0
    epochs_without_improvement = 0

    if args.resume:

        if not LAST_CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"No checkpoint found at "
                f"{LAST_CHECKPOINT_PATH}"
            )

        checkpoint = torch.load(
            LAST_CHECKPOINT_PATH,
            map_location=DEVICE,
            weights_only=False,
        )

        expected_labels = checkpoint.get(
            "finding_labels",
            FINDING_LABELS,
        )

        if expected_labels != FINDING_LABELS:
            raise ValueError(
                "Checkpoint finding labels do not match config.py."
            )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(
                checkpoint["scaler_state_dict"]
            )

        start_epoch = (
            int(checkpoint["completed_epoch"]) + 1
        )

        best_combined_auc = float(
            checkpoint.get(
                "best_combined_auc",
                -1.0,
            )
        )

        epochs_without_improvement = int(
            checkpoint.get(
                "epochs_without_improvement",
                0,
            )
        )

        print(f"Resuming at epoch {start_epoch}.")

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):

        model.train()
        batch_losses: list[float] = []

        for batch_number, batch in enumerate(
            train_loader,
            start=1,
        ):

            (
                images,
                normal_targets,
                normal_masks,
                finding_targets,
                finding_masks,
                _paths,
            ) = batch

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            normal_targets = normal_targets.to(
                DEVICE,
                non_blocking=True,
            )

            normal_masks = normal_masks.to(
                DEVICE,
                non_blocking=True,
            )

            finding_targets = finding_targets.to(
                DEVICE,
                non_blocking=True,
            )

            finding_masks = finding_masks.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                "cuda",
                enabled=torch.cuda.is_available(),
            ):
                normal_logits, finding_logits = model(images)

                normal_loss = normal_loss_function(
                    normal_logits,
                    normal_targets,
                    normal_masks,
                )

                # Do not teach the finding head to call abnormalities on
                # confirmed normal studies.
                effective_finding_mask = finding_masks

                finding_loss = finding_loss_function(
                    finding_logits,
                    finding_targets,
                    effective_finding_mask,
                )

                total_loss = normal_loss + finding_loss

            scaler.scale(total_loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            scaler.step(optimizer)
            scaler.update()

            batch_losses.append(
                float(total_loss.item())
            )

            if (
                batch_number % 100 == 0
                or batch_number == len(train_loader)
            ):
                print(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"Batch {batch_number}/"
                    f"{len(train_loader)} | "
                    f"Loss {total_loss.item():.4f}",
                    flush=True,
                )

        (
            validation_loss,
            normal_auc,
            findings_macro_auc,
            combined_auc,
            per_finding_auc,
        ) = validation_pass(
            model,
            validation_loader,
            normal_loss_function,
            finding_loss_function,
        )

        scheduler.step(combined_auc)

        training_loss = float(
            np.mean(batch_losses)
        )

        learning_rate = optimizer.param_groups[0]["lr"]

        print("\n" + "=" * 66)
        print(f"Epoch {epoch} complete")
        print(f"Training loss:             {training_loss:.4f}")
        print(f"Validation loss:           {validation_loss:.4f}")

        if normal_auc is None:
            print("Normal-gate AUC:            unavailable")
        else:
            print(f"Normal-gate AUC:            {normal_auc:.4f}")

        print(
            f"Finding-head macro AUC:    "
            f"{findings_macro_auc:.4f}"
        )

        print(
            f"Combined validation AUC:   "
            f"{combined_auc:.4f}"
        )

        print(f"Learning rate:             {learning_rate:.2e}")
        print("-" * 66)

        for label, auc in per_finding_auc.items():
            print(f"{label:28s} AUC={auc:.4f}")

        print("=" * 66 + "\n")

        improved = (
            combined_auc > best_combined_auc + 1e-4
        )

        if improved:
            best_combined_auc = combined_auc
            epochs_without_improvement = 0

            save_best_model(
                model=model,
                combined_auc=combined_auc,
                normal_auc=normal_auc,
                findings_macro_auc=findings_macro_auc,
            )

            print(
                f"Saved new best V2 model to "
                f"{MODEL_PATH}"
            )

        else:
            epochs_without_improvement += 1

            print(
                "No improvement for "
                f"{epochs_without_improvement} epoch(s)."
            )

        save_checkpoint(
            path=CHECKPOINT_DIR / f"epoch_{epoch}.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            completed_epoch=epoch,
            best_combined_auc=best_combined_auc,
            epochs_without_improvement=(
                epochs_without_improvement
            ),
        )

        save_checkpoint(
            path=LAST_CHECKPOINT_PATH,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            completed_epoch=epoch,
            best_combined_auc=best_combined_auc,
            epochs_without_improvement=(
                epochs_without_improvement
            ),
        )

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping after "
                f"{args.patience} epochs without improvement."
            )
            break

    print(
        "Training finished. "
        f"Best combined validation AUC: "
        f"{best_combined_auc:.4f}"
    )

    print("Next run: python evaluate.py")


if __name__ == "__main__":
    main()