from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import densenet121

from bone_config import BATCH_SIZE, IMAGE_SIZE, MURA_ROOT, NUM_WORKERS, VALID_IMAGE_PATHS
from dataset_mura import MURADataset
from train_body_part import BODY_MODEL_PATH, CLASS_TO_INDEX

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_PATH = Path(__file__).resolve().parent / "models" / "mura_body_part_metrics.json"


class BodyPartDataset(MURADataset):
    def __getitem__(self, index: int):
        image, _label, body_part, image_path = super().__getitem__(index)
        target = torch.tensor(CLASS_TO_INDEX[body_part], dtype=torch.long)
        return image, target, image_path


def main():
    checkpoint = torch.load(BODY_MODEL_PATH, map_location=DEVICE, weights_only=False)
    class_names = checkpoint["class_names"]

    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = BodyPartDataset(VALID_IMAGE_PATHS, MURA_ROOT, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS,
                        pin_memory=torch.cuda.is_available())

    targets_all, predictions_all = [], []

    with torch.no_grad():
        for images, targets, _paths in loader:
            images = images.to(DEVICE, non_blocking=True)
            predictions = torch.argmax(model(images), dim=1)
            targets_all.append(targets.numpy())
            predictions_all.append(predictions.cpu().numpy())

    targets = np.concatenate(targets_all)
    predictions = np.concatenate(predictions_all)

    output = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "class_names": class_names,
        "classification_report": classification_report(
            targets, predictions, target_names=class_names,
            output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(targets, predictions).tolist(),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"\nSaved metrics to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
