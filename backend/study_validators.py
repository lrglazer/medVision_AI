from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torchvision import transforms

from ai.shared.study_validator import build_validator_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ValidatorUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class StudyValidation:
    accepted: bool
    score: float
    threshold: float
    display_score: str
    validation_auc: float
    validation_accuracy: float
    validation_specificity: float


class StudyValidator:
    def __init__(self, checkpoint_path: Path) -> None:
        self.checkpoint_path = checkpoint_path
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        if not self.checkpoint_path.exists():
            raise ValidatorUnavailableError(
                f"Validator checkpoint not found: {self.checkpoint_path}"
            )
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=DEVICE,
            weights_only=False,
        )
        self.image_size = int(checkpoint.get("image_size", 224))
        self.threshold = float(checkpoint.get("threshold", 0.5))
        self.validation_auc = float(checkpoint.get("validation_auc", 0.0))
        self.validation_accuracy = float(checkpoint.get("validation_accuracy", 0.0))
        self.validation_specificity = float(checkpoint.get("validation_specificity", 0.0))

        self.model = build_validator_model(pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(DEVICE).eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        self._loaded = True

    def evaluate(self, image: Image.Image) -> StudyValidation:
        self._load()
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            score = float(torch.sigmoid(self.model(tensor)).item())
        return StudyValidation(
            accepted=score >= self.threshold,
            score=score,
            threshold=self.threshold,
            display_score=f"{score:.1%}",
            validation_auc=self.validation_auc,
            validation_accuracy=self.validation_accuracy,
            validation_specificity=self.validation_specificity,
        )


class BoneClipValidator:
    POSITIVE_PROMPTS = [
        "a medical x-ray radiograph of a human bone",
        "a musculoskeletal radiograph",
        "an orthopedic x-ray image",
        "an x-ray of a human hand wrist finger elbow forearm humerus or shoulder",
        "a grayscale diagnostic bone radiograph",
    ]

    NEGATIVE_PROMPTS = [
        "a chemistry diagram or molecular structure",
        "a screenshot containing text or graphics",
        "a normal photograph",
        "a drawing illustration chart or infographic",
        "food cake or a household object",
        "a pet animal or person photographed with a camera",
        "a chest x-ray radiograph",
        "a CT scan MRI ultrasound or other non bone x-ray image",
    ]

    def __init__(
        self,
        threshold: float = 0.62,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
    ) -> None:
        self.threshold = threshold
        self.model_name = model_name
        self.pretrained = pretrained
        self._loaded = False
        self.model = None
        self.preprocess = None
        self.positive_text_features: Optional[torch.Tensor] = None
        self.negative_text_features: Optional[torch.Tensor] = None

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            import open_clip
        except ImportError as error:
            raise ValidatorUnavailableError(
                "OpenCLIP is not installed. Run: pip install open_clip_torch"
            ) from error

        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=DEVICE,
            )
            tokenizer = open_clip.get_tokenizer(self.model_name)
            model = model.eval()

            with torch.no_grad():
                pos_tokens = tokenizer(self.POSITIVE_PROMPTS).to(DEVICE)
                neg_tokens = tokenizer(self.NEGATIVE_PROMPTS).to(DEVICE)
                pos = model.encode_text(pos_tokens)
                neg = model.encode_text(neg_tokens)
                pos = pos / pos.norm(dim=-1, keepdim=True)
                neg = neg / neg.norm(dim=-1, keepdim=True)
                self.positive_text_features = pos.mean(dim=0, keepdim=True)
                self.positive_text_features = self.positive_text_features / self.positive_text_features.norm(dim=-1, keepdim=True)
                self.negative_text_features = neg.mean(dim=0, keepdim=True)
                self.negative_text_features = self.negative_text_features / self.negative_text_features.norm(dim=-1, keepdim=True)

            self.model = model
            self.preprocess = preprocess
            self._loaded = True
        except Exception as error:
            raise ValidatorUnavailableError(
                "OpenCLIP could not load its pretrained model. The first run may require internet access to download weights."
            ) from error

    def evaluate(self, image: Image.Image) -> StudyValidation:
        self._load()
        assert self.model is not None
        assert self.preprocess is not None
        assert self.positive_text_features is not None
        assert self.negative_text_features is not None

        image_tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            positive_similarity = (image_features @ self.positive_text_features.T).squeeze()
            negative_similarity = (image_features @ self.negative_text_features.T).squeeze()
            logits = torch.stack([negative_similarity, positive_similarity]) * 100.0
            score = float(torch.softmax(logits, dim=0)[1].item())

        return StudyValidation(
            accepted=score >= self.threshold,
            score=score,
            threshold=self.threshold,
            display_score=f"{score:.1%}",
            validation_auc=0.0,
            validation_accuracy=0.0,
            validation_specificity=0.0,
        )


CHEST_VALIDATOR = StudyValidator(
    PROJECT_ROOT / "ai" / "chest" / "models" / "best_chest_validator.pt"
)

BONE_VALIDATOR = BoneClipValidator(threshold=0.62)