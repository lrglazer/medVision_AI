from pathlib import Path

CHEXPERT_ROOT = Path("../../datasets/CheXpert-v1.0-small")
TRAIN_CSV = CHEXPERT_ROOT / "train.csv"
VALID_CSV = CHEXPERT_ROOT / "valid.csv"

OUTPUT_DIR = Path("models")
MODEL_PATH = OUTPUT_DIR / "best_chexpert_multilabel.pt"
THRESHOLDS_PATH = OUTPUT_DIR / "chexpert_thresholds.json"
METRICS_PATH = OUTPUT_DIR / "chexpert_validation_metrics.csv"

IMAGE_SIZE = 224

LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

UNCERTAINTY_POLICY = {
    "No Finding": "zeros",
    "Enlarged Cardiomediastinum": "ignore",
    "Cardiomegaly": "zeros",
    "Lung Opacity": "ignore",
    "Lung Lesion": "ignore",
    "Edema": "ones",
    "Consolidation": "zeros",
    "Pneumonia": "ignore",
    "Atelectasis": "ones",
    "Pneumothorax": "ignore",
    "Pleural Effusion": "zeros",
    "Pleural Other": "ignore",
    "Fracture": "ignore",
    "Support Devices": "ignore",
}
