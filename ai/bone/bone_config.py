from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MURA_ROOT = PROJECT_ROOT / "datasets" / "MURA-v1.1"

TRAIN_IMAGE_PATHS = MURA_ROOT / "train_image_paths.csv"
VALID_IMAGE_PATHS = MURA_ROOT / "valid_image_paths.csv"

MODELS_DIR = Path(__file__).resolve().parent / "models"
CHECKPOINTS_DIR = Path(__file__).resolve().parent / "checkpoints"

BEST_MODEL_PATH = MODELS_DIR / "best_mura_abnormality.pt"
LAST_CHECKPOINT_PATH = CHECKPOINTS_DIR / "last.pt"
METRICS_PATH = MODELS_DIR / "mura_validation_metrics.json"
THRESHOLD_PATH = MODELS_DIR / "mura_threshold.json"

IMAGE_SIZE = 320
BATCH_SIZE = 24
TOTAL_EPOCHS = 5
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0

BODY_PARTS = [
    "XR_ELBOW",
    "XR_FINGER",
    "XR_FOREARM",
    "XR_HAND",
    "XR_HUMERUS",
    "XR_SHOULDER",
    "XR_WRIST",
]
