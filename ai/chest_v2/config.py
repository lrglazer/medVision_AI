from pathlib import Path

# CheXpert dataset paths
CHEXPERT_ROOT = Path("../../datasets/CheXpert-v1.0-small")
TRAIN_CSV = CHEXPERT_ROOT / "train.csv"
VALID_CSV = CHEXPERT_ROOT / "valid.csv"

# Chest V2 output paths
# These are separate so the original chest model is not overwritten.
OUTPUT_DIR = Path("models_v2")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LAST_CHECKPOINT_PATH = CHECKPOINT_DIR / "last.pt"

MODEL_PATH = OUTPUT_DIR / "best_chest_v2.pt"
THRESHOLDS_PATH = OUTPUT_DIR / "chest_v2_thresholds.json"
METRICS_PATH = OUTPUT_DIR / "chest_v2_validation_metrics.csv"

IMAGE_SIZE = 224

# Separate normal-versus-abnormal gate
NORMAL_LABEL = "No Finding"

# Clinically relevant abnormal findings
FINDING_LABELS = [
    "Cardiomegaly",
    "Lung Opacity",
    "Pneumonia",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
    "Pneumothorax",
    "Atelectasis",
]

# Columns that must exist in the CheXpert CSV files
CSV_LABELS = [
    NORMAL_LABEL,
    *FINDING_LABELS,
]

# How uncertain CheXpert labels (-1) should be handled
#
# "ones"   = treat uncertain as positive
# "zeros"  = treat uncertain as negative
# "ignore" = exclude from training loss and evaluation
UNCERTAINTY_POLICY = {
    NORMAL_LABEL: "zeros",
    "Cardiomegaly": "zeros",
    "Lung Opacity": "ignore",
    "Pneumonia": "ignore",
    "Consolidation": "zeros",
    "Edema": "ones",
    "Pleural Effusion": "zeros",
    "Pneumothorax": "ignore",
    "Atelectasis": "ones",
}

# Evaluation and threshold calibration settings
MIN_SPECIFICITY = 0.85
NORMAL_GATE_MIN_SPECIFICITY = 0.90
MIN_POSITIVES_FOR_CALIBRATION = 10