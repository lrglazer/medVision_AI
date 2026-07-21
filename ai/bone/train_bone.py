from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import DenseNet121_Weights, densenet121

from bone_config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    CHECKPOINTS_DIR,
    IMAGE_SIZE,
    LAST_CHECKPOINT_PATH,
    LEARNING_RATE,
    MURA_ROOT,
    NUM_WORKERS,
    TOTAL_EPOCHS,
    TRAIN_IMAGE_PATHS,
    VALID_IMAGE_PATHS,
    WEIGHT_DECAY,
)
from dataset_mura import MURADataset


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the MedVision MURA Bone Specialist."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoints/last.pt.",
    )
    return parser.parse_args()


def build_transforms():
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(7),
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

    valid_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return train_transform, valid_transform


def build_model() -> nn.Module:
    model = densenet121(
        weights=DenseNet121_Weights.DEFAULT
    )
    model.classifier = nn.Linear(
        model.classifier.in_features,
        1,
    )
    return model.to(DEVICE)


def compute_pos_weight(
    dataset: MURADataset,
) -> torch.Tensor:
    positives = float(dataset.frame["label"].sum())
    negatives = float(len(dataset.frame) - positives)

    weight = negatives / max(positives, 1.0)

    print()
    print(f"Positive studies/images: {int(positives):,}")
    print(f"Negative studies/images: {int(negatives):,}")
    print(f"Positive-class weight: {weight:.4f}")

    return torch.tensor(
        [weight],
        dtype=torch.float32,
        device=DEVICE,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
) -> tuple[float, float]:
    model.eval()

    losses: list[float] = []
    all_targets: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels, _body_parts, _paths in loader:
            images = images.to(
                DEVICE,
                non_blocking=True,
            )
            labels = labels.to(
                DEVICE,
                non_blocking=True,
            ).unsqueeze(1)

            with torch.amp.autocast(
                "cuda",
                enabled=torch.cuda.is_available(),
            ):
                logits = model(images)
                loss = loss_function(logits, labels)

            scores = torch.sigmoid(logits)

            losses.append(float(loss.item()))
            all_targets.append(labels.cpu().numpy())
            all_scores.append(scores.cpu().numpy())

    targets = np.concatenate(all_targets).ravel()
    scores = np.concatenate(all_scores).ravel()

    auc = float(roc_auc_score(targets, scores))
    validation_loss = float(np.mean(losses))

    return validation_loss, auc


def save_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    completed_epoch: int,
    best_auc: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "completed_epoch": completed_epoch,
            "best_auc": best_auc,
            "architecture": "densenet121",
            "image_size": IMAGE_SIZE,
            "task": "mura_abnormality",
        },
        path,
    )


def save_best_model(
    model: nn.Module,
    validation_auc: float,
) -> None:
    BEST_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "validation_auc": validation_auc,
            "architecture": "densenet121",
            "image_size": IMAGE_SIZE,
            "task": "mura_abnormality",
        },
        BEST_MODEL_PATH,
    )


def main() -> None:
    args = parse_args()

    print("Using device:", DEVICE)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    CHECKPOINTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    BEST_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_transform, valid_transform = build_transforms()

    train_dataset = MURADataset(
        csv_file=TRAIN_IMAGE_PATHS,
        mura_root=MURA_ROOT,
        transform=train_transform,
    )
    valid_dataset = MURADataset(
        csv_file=VALID_IMAGE_PATHS,
        mura_root=MURA_ROOT,
        transform=valid_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model()
    pos_weight = compute_pos_weight(train_dataset)

    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=torch.cuda.is_available(),
    )

    start_epoch = 1
    best_auc = 0.0

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

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(
                checkpoint["scaler_state_dict"]
            )

        completed_epoch = int(
            checkpoint["completed_epoch"]
        )
        start_epoch = completed_epoch + 1
        best_auc = float(
            checkpoint.get("best_auc", 0.0)
        )

        print(
            f"Resuming after Epoch {completed_epoch}. "
            f"Next epoch: {start_epoch}."
        )

    for epoch in range(
        start_epoch,
        TOTAL_EPOCHS + 1,
    ):
        model.train()
        running_losses: list[float] = []

        for batch_number, (
            images,
            labels,
            _body_parts,
            _paths,
        ) in enumerate(
            train_loader,
            start=1,
        ):
            images = images.to(
                DEVICE,
                non_blocking=True,
            )
            labels = labels.to(
                DEVICE,
                non_blocking=True,
            ).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                "cuda",
                enabled=torch.cuda.is_available(),
            ):
                logits = model(images)
                loss = loss_function(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_losses.append(float(loss.item()))

            if batch_number % 100 == 0:
                print(
                    f"Epoch {epoch}/{TOTAL_EPOCHS} | "
                    f"Batch {batch_number}/{len(train_loader)} | "
                    f"Loss {loss.item():.4f}"
                )

        train_loss = float(
            np.mean(running_losses)
        )
        valid_loss, valid_auc = evaluate(
            model,
            valid_loader,
            loss_function,
        )

        print()
        print(f"Epoch {epoch} complete")
        print(f"Training loss:   {train_loss:.4f}")
        print(f"Validation loss: {valid_loss:.4f}")
        print(f"Validation AUC:  {valid_auc:.4f}")
        print()

        updated_best_auc = max(
            best_auc,
            valid_auc,
        )

        epoch_path = (
            CHECKPOINTS_DIR / f"epoch_{epoch}.pt"
        )

        save_checkpoint(
            path=epoch_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            completed_epoch=epoch,
            best_auc=updated_best_auc,
        )
        save_checkpoint(
            path=LAST_CHECKPOINT_PATH,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            completed_epoch=epoch,
            best_auc=updated_best_auc,
        )

        if valid_auc > best_auc:
            best_auc = valid_auc
            save_best_model(
                model,
                validation_auc=best_auc,
            )
            print(
                f"Saved improved best model to "
                f"{BEST_MODEL_PATH}"
            )

    print()
    print("Training finished.")
    print(f"Best validation AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
