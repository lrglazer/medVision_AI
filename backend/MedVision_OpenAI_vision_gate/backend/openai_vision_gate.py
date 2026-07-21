from __future__ import annotations

import base64
import io
import os
from enum import Enum
from typing import Literal

from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, Field


class StudyType(str, Enum):
    CHEST_XRAY = "chest_xray"
    BONE_XRAY = "bone_xray"
    OTHER = "other"


class VisionGateResult(BaseModel):
    study_type: StudyType
    accepted: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    detected_region: str | None = None


class VisionGateError(RuntimeError):
    pass


def _image_to_data_url(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    rgb.thumbnail((1200, 1200))

    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


class OpenAIVisionGate:
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise VisionGateError(
                "OPENAI_API_KEY is not configured for the study validator."
            )

        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv("OPENAI_VISION_GATE_MODEL", "gpt-5.6")

    def classify(
        self,
        image: Image.Image,
        expected: Literal["chest", "bone"],
    ) -> VisionGateResult:
        expected_description = (
            "a frontal or lateral chest radiograph"
            if expected == "chest"
            else (
                "a musculoskeletal radiograph of an elbow, finger, forearm, "
                "hand, humerus, shoulder, or wrist"
            )
        )

        prompt = f"""
Classify whether the uploaded image is suitable for the MedVision {expected}
specialist.

Accept only when the image is clearly {expected_description}.

Reject:
- ordinary photographs
- food, people, pets, objects, artwork, memes, or screenshots
- charts, documents, and user-interface captures
- CT, MRI, ultrasound, pathology, or non-radiograph medical images
- radiographs of the wrong anatomical study
- images where the modality or anatomy cannot be determined confidently

Return:
- study_type: chest_xray, bone_xray, or other
- accepted: true only when it matches the requested specialist
- confidence: confidence in this routing decision from 0 to 1
- reason: one brief user-facing sentence
- detected_region: the visible anatomical region when identifiable

This is only study routing. Do not diagnose disease or describe pathology.
""".strip()

        response = self.client.responses.parse(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict medical-image study router. "
                        "When uncertain, reject the upload."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": _image_to_data_url(image),
                            "detail": "low",
                        },
                    ],
                },
            ],
            text_format=VisionGateResult,
        )

        result = response.output_parsed
        if result is None:
            raise VisionGateError("The vision validator returned no result.")

        expected_type = (
            StudyType.CHEST_XRAY
            if expected == "chest"
            else StudyType.BONE_XRAY
        )

        # Enforce the requested route even if the model's accepted field and
        # study_type ever disagree.
        result.accepted = bool(
            result.accepted
            and result.study_type == expected_type
            and result.confidence >= 0.80
        )
        return result


_GATE: OpenAIVisionGate | None = None


def get_vision_gate() -> OpenAIVisionGate:
    global _GATE
    if _GATE is None:
        _GATE = OpenAIVisionGate()
    return _GATE
