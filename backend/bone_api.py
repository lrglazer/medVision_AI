from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
from torchvision.models import densenet121

from backend.study_validators import (
    BONE_VALIDATOR,
    ValidatorUnavailableError,
)

router = APIRouter(prefix="/api/bone", tags=["Bone Specialist"])

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "ai" / "bone" / "models"
ABNORMALITY_MODEL_PATH = MODELS_DIR / "best_mura_abnormality.pt"
BODY_PART_MODEL_PATH = MODELS_DIR / "best_mura_body_part.pt"
THRESHOLD_PATH = MODELS_DIR / "mura_threshold.json"
ABNORMALITY_METRICS_PATH = MODELS_DIR / "mura_validation_metrics.json"
BODY_PART_METRICS_PATH = MODELS_DIR / "mura_body_part_metrics.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_IMAGE_SIZE = 320


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_model(outputs: int) -> nn.Module:
    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, outputs)
    return model


def load_models():
    abnormality_checkpoint = torch.load(
        ABNORMALITY_MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )
    body_part_checkpoint = torch.load(
        BODY_PART_MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    abnormality_model = build_model(1)
    abnormality_model.load_state_dict(
        abnormality_checkpoint["model_state_dict"]
    )
    abnormality_model = abnormality_model.to(DEVICE).eval()

    class_names = body_part_checkpoint["class_names"]
    body_part_model = build_model(len(class_names))
    body_part_model.load_state_dict(
        body_part_checkpoint["model_state_dict"]
    )
    body_part_model = body_part_model.to(DEVICE).eval()

    return (
        abnormality_model,
        body_part_model,
        class_names,
        int(abnormality_checkpoint.get("image_size", DEFAULT_IMAGE_SIZE)),
        float(abnormality_checkpoint.get("validation_auc", 0.0)),
        float(body_part_checkpoint.get("validation_accuracy", 0.0)),
    )


(
    abnormality_model,
    body_part_model,
    BODY_PART_CLASSES,
    IMAGE_SIZE,
    ABNORMALITY_AUC,
    BODY_PART_ACCURACY,
) = load_models()

THRESHOLD_CONFIG = load_json(THRESHOLD_PATH)
ABNORMALITY_METRICS = load_json(ABNORMALITY_METRICS_PATH)
BODY_PART_METRICS = load_json(BODY_PART_METRICS_PATH)

image_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


class BinaryOutputTarget:
    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        return model_output[0]


def prepare_image(image: Image.Image):
    rgb = image.convert("RGB")
    resized = rgb.resize((IMAGE_SIZE, IMAGE_SIZE))
    array = np.asarray(resized).astype(np.float32) / 255.0
    tensor = image_transform(rgb).unsqueeze(0).to(DEVICE)
    return resized, array, tensor


def detect_body_part(tensor: torch.Tensor):
    with torch.no_grad():
        probabilities = torch.softmax(body_part_model(tensor), dim=1)[0]

    top_score, top_index = torch.max(probabilities, dim=0)
    all_scores = [
        {
            "name": name,
            "score": float(probabilities[index].item()),
            "display_score": f"{float(probabilities[index].item()):.1%}",
        }
        for index, name in enumerate(BODY_PART_CLASSES)
    ]
    all_scores.sort(key=lambda item: item["score"], reverse=True)

    return (
        BODY_PART_CLASSES[int(top_index.item())],
        float(top_score.item()),
        all_scores,
    )


def predict_abnormality(tensor: torch.Tensor) -> float:
    with torch.no_grad():
        return float(torch.sigmoid(abnormality_model(tensor)).item())


def classify(score: float):
    threshold = float(THRESHOLD_CONFIG.get("threshold", 0.5))
    negative_max = float(
        THRESHOLD_CONFIG.get("negative_max", max(0.0, threshold - 0.10))
    )
    positive_min = float(
        THRESHOLD_CONFIG.get("positive_min", min(1.0, threshold + 0.10))
    )

    if score <= negative_max:
        status = "Negative model finding"
    elif score >= positive_min:
        status = "Positive model finding"
    else:
        status = "Indeterminate"

    return status, threshold


def make_gradcam(array: np.ndarray, tensor: torch.Tensor) -> np.ndarray:
    with GradCAM(
        model=abnormality_model,
        target_layers=[abnormality_model.features.denseblock4],
    ) as cam:
        heatmap = cam(
            input_tensor=tensor,
            targets=[BinaryOutputTarget()],
        )[0]

    return show_cam_on_image(array, heatmap, use_rgb=True)


def encode_png(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def analyze(image: Image.Image):
    _, array, tensor = prepare_image(image)
    body_part, body_part_score, body_part_predictions = detect_body_part(tensor)
    abnormality_score = predict_abnormality(tensor)
    status, threshold = classify(abnormality_score)
    overlay = np.asarray(image.convert("RGB"))

    if status == "Positive model finding":
        interpretation = (
            f"The model identified imaging features associated with an abnormal "
            f"{body_part.lower()} radiograph."
        )
    elif status == "Indeterminate":
        interpretation = (
            f"The abnormality output for this {body_part.lower()} radiograph "
            f"is indeterminate."
        )
    else:
        interpretation = (
            f"The model did not identify a high abnormality output for this "
            f"{body_part.lower()} radiograph."
        )

    return {
        "body_part": body_part,
        "body_part_score": body_part_score,
        "body_part_predictions": body_part_predictions,
        "abnormality_score": abnormality_score,
        "status": status,
        "threshold": threshold,
        "overlay": overlay,
        "interpretation": interpretation,
    }


@router.get("/health")
def bone_health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "body_part_accuracy": BODY_PART_ACCURACY,
        "abnormality_auc": ABNORMALITY_AUC,
        "classes": BODY_PART_CLASSES,
    }


@router.post("/predict")
async def predict_bone(file: UploadFile = File(...)):
    
    if file.content_type not in {"image/png", "image/jpeg"}:
        raise HTTPException(status_code=400, detail="Upload a PNG or JPEG X-ray.")

    try:
        print("BONE: request received", flush=True)

        image = Image.open(io.BytesIO(await file.read()))

        print("BONE: validator starting", flush=True)
        validation = BONE_VALIDATOR.evaluate(image)
        print("BONE: validator finished", flush=True)

        if not validation.accepted:
            raise HTTPException(
                status_code=422,
                detail=(
                    "This image does not appear to be a supported musculoskeletal X-ray. "
                    f"Validator score: {validation.display_score}. "
                    "Please upload an elbow, finger, forearm, hand, humerus, "
                    "shoulder, or wrist radiograph."
                ),
            )

        print("BONE: analysis starting", flush=True)
        result = analyze(image)
        print("BONE: analysis finished", flush=True)

        print("BONE: returning response", flush=True)

        return {
            "filename": file.filename,
            "input_validation": {
                "accepted": validation.accepted,
                "score": validation.score,
                "display_score": validation.display_score,
                "threshold": validation.threshold,
                "validator_auc": validation.validation_auc,
                "validator_accuracy": validation.validation_accuracy,
                "validator_specificity": validation.validation_specificity,
            },
            "study": {
                "type": f"{result['body_part']} radiograph",
                "body_part": result["body_part"],
                "body_part_score": result["body_part_score"],
                "body_part_display_score": f"{result['body_part_score']:.1%}",
                "specialist": "Bone Specialist",
                "model": "DenseNet-121",
                "validation_auc": ABNORMALITY_AUC,
                "body_part_validation_accuracy": BODY_PART_ACCURACY,
            },
            "body_part_predictions": result["body_part_predictions"],
            "abnormality": {
                "score": result["abnormality_score"],
                "display_score": f"{result['abnormality_score']:.1%}",
                "status": result["status"],
                "threshold": result["threshold"],
                "display_threshold": f"{result['threshold']:.1%}",
            },
            "interpretation": result["interpretation"],
            "gradcam": {
                "image_base64": encode_png(result["overlay"]),
                "explanation": (
                    "The heatmap shows image regions that influenced the "
                    "abnormality output. It is not lesion segmentation."
                ),
            },
            "metrics": {
                "auc": ABNORMALITY_METRICS.get("auc", ABNORMALITY_AUC),
                "precision": ABNORMALITY_METRICS.get("precision"),
                "sensitivity": ABNORMALITY_METRICS.get("sensitivity"),
                "specificity": ABNORMALITY_METRICS.get("specificity"),
                "f1": ABNORMALITY_METRICS.get("f1"),
                "body_part_accuracy": BODY_PART_METRICS.get(
                    "accuracy", BODY_PART_ACCURACY
                ),
            },
            "disclaimer": (
                "Educational research use only. This is an abnormality model, "
                "not a fracture diagnosis."
            ),
        }

    except HTTPException:
        raise
    except ValidatorUnavailableError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "The bone input validator is not trained or available. "
                "The medical model was not run."
            ),
        ) from error
    except UnidentifiedImageError as error:
        raise HTTPException(status_code=400, detail="Invalid image file.") from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error