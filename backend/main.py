from __future__ import annotations

import base64
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as ReportLabImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from torchvision import transforms
from torchvision.models import densenet121

from backend.bone_api import router as bone_router
from backend.study_validators import (
    CHEST_VALIDATOR,
    ValidatorUnavailableError,
)


app = FastAPI(title="MedVision API", version="2.1.0")

app.include_router(bone_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://medvisionai.app",
        "https://www.medvisionai.app",
        "https://med-vision-ai-htvv-nu.vercel.app",
    ],
    allow_origin_regex=r"https://med-vision-.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "ai"
    / "chest_v2"
    / "models_v2"
    / "best_chest_v2.pt"
)

THRESHOLDS_PATH = (
    PROJECT_ROOT
    / "ai"
    / "chest_v2"
    / "models_v2"
    / "chest_v2_thresholds.json"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224

NORMAL_LABEL = "No Finding"

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

REPORT_EXCLUDED_LABELS: set[str] = set()

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


class ChestV2Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        backbone = densenet121(weights=None)
        feature_count = backbone.classifier.in_features

        self.features = backbone.features
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


class FindingHeadWrapper(nn.Module):
    """
    Grad-CAM needs one tensor output, so this wrapper exposes only
    the eight abnormal-finding logits.
    """

    def __init__(self, chest_model: ChestV2Model) -> None:
        super().__init__()
        self.chest_model = chest_model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        _, finding_logits = self.chest_model(images)
        return finding_logits


class MultiLabelOutputTarget:
    def __init__(self, category_index: int):
        self.category_index = category_index

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        return model_output[self.category_index]


def load_thresholds() -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
]:
    if not THRESHOLDS_PATH.exists():
        raise FileNotFoundError(
            f"Threshold file not found at {THRESHOLDS_PATH}"
        )

    with THRESHOLDS_PATH.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    normal_config = raw.get(
        "normal_gate",
        {},
    ).get(NORMAL_LABEL)

    finding_configs = raw.get("findings")

    if not isinstance(normal_config, dict):
        raise ValueError(
            "Threshold file is missing normal_gate -> No Finding."
        )

    if not isinstance(finding_configs, dict):
        raise ValueError(
            "Threshold file is missing finding thresholds."
        )

    missing = set(FINDING_LABELS) - set(finding_configs)

    if missing:
        raise ValueError(
            f"Threshold file is missing labels: {sorted(missing)}"
        )

    return normal_config, finding_configs


def load_model() -> tuple[
    ChestV2Model,
    float,
    float,
    float,
]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    checkpoint_labels = checkpoint.get(
        "finding_labels",
        FINDING_LABELS,
    )

    if checkpoint_labels != FINDING_LABELS:
        raise ValueError(
            "Checkpoint finding labels do not match the backend."
        )

    model = ChestV2Model()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    combined_auc = float(
        checkpoint.get(
            "validation_combined_auc",
            0.0,
        )
    )

    normal_auc = float(
        checkpoint.get(
            "validation_normal_auc",
            0.0,
        )
        or 0.0
    )

    findings_macro_auc = float(
        checkpoint.get(
            "validation_findings_macro_auc",
            0.0,
        )
    )

    return (
        model,
        combined_auc,
        normal_auc,
        findings_macro_auc,
    )


(
    model,
    VALIDATION_COMBINED_AUC,
    VALIDATION_NORMAL_AUC,
    VALIDATION_FINDINGS_MACRO_AUC,
) = load_model()

NORMAL_THRESHOLD_CONFIG, THRESHOLDS = load_thresholds()

gradcam_model = FindingHeadWrapper(model)
gradcam_model = gradcam_model.to(DEVICE)
gradcam_model.eval()


def prepare_image(
    image: Image.Image,
) -> tuple[np.ndarray, torch.Tensor, Image.Image]:
    rgb_image = image.convert("RGB")
    resized_image = rgb_image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_array = (
        np.asarray(resized_image).astype(np.float32) / 255.0
    )
    input_tensor = image_transform(rgb_image).unsqueeze(0).to(DEVICE)
    return image_array, input_tensor, resized_image


def get_probabilities(
    input_tensor: torch.Tensor,
) -> tuple[float, np.ndarray]:

    with torch.no_grad():
        normal_logits, finding_logits = model(input_tensor)

        normal_probability = torch.sigmoid(normal_logits)
        finding_probabilities = torch.sigmoid(finding_logits)

    return (
        float(normal_probability.squeeze(0).item()),
        finding_probabilities.squeeze(0).cpu().numpy(),
    )


def classify_score(label: str, score: float) -> tuple[str, float]:
    config = THRESHOLDS.get(label, {})
    threshold = float(config.get("threshold", 0.5))
    negative_max = float(
        config.get("negative_max", max(0.0, threshold - 0.10))
    )
    positive_min = float(
        config.get("positive_min", min(1.0, threshold + 0.10))
    )

    if score <= negative_max:
        status = "Negative model finding"
    elif score >= positive_min:
        status = "Positive model finding"
    else:
        status = "Indeterminate"

    return status, threshold


def classify_normal_gate(score: float) -> dict[str, Any]:

    threshold = float(
        NORMAL_THRESHOLD_CONFIG.get(
            "threshold",
            0.5,
        )
    )

    negative_max = float(
        NORMAL_THRESHOLD_CONFIG.get(
            "negative_max",
            max(0.0, threshold - 0.05),
        )
    )

    positive_min = float(
        NORMAL_THRESHOLD_CONFIG.get(
            "positive_min",
            min(1.0, threshold + 0.05),
        )
    )

    if score >= positive_min:
        status = "Normal model finding"
    elif score <= negative_max:
        status = "Abnormal model finding"
    else:
        status = "Indeterminate"

    return {
        "name": NORMAL_LABEL,
        "score": score,
        "display_score": f"{score:.1%}",
        "status": status,
        "threshold": threshold,
        "display_threshold": f"{threshold:.1%}",
        "negative_max": negative_max,
        "positive_min": positive_min,
    }


def build_findings(
    probabilities: np.ndarray,
) -> list[dict[str, Any]]:

    if len(probabilities) != len(FINDING_LABELS):
        raise ValueError(
            "Model output count does not match FINDING_LABELS."
        )

    findings: list[dict[str, Any]] = []

    for index, label in enumerate(FINDING_LABELS):
        score = float(probabilities[index])
        status, threshold = classify_score(label, score)

        findings.append(
            {
                "name": label,
                "score": score,
                "display_score": f"{score:.1%}",
                "status": status,
                "threshold": threshold,
                "display_threshold": f"{threshold:.1%}",
            }
        )

    findings.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return findings


def build_report(
    findings: list[dict[str, Any]],
    normal_gate: dict[str, Any],
) -> tuple[str, list[str], str]:
    report_findings = findings
    positive = [
        finding
        for finding in report_findings
        if finding["status"] == "Positive model finding"
    ]
    indeterminate = [
        finding
        for finding in report_findings
        if finding["status"] == "Indeterminate"
    ]
    negative = [
        finding
        for finding in report_findings
        if finding["status"] == "Negative model finding"
    ]

    sentences: list[str] = []

    if positive:
        sentences.append(
            "Elevated model outputs are present for "
            + ", ".join(finding["name"] for finding in positive[:5])
            + "."
        )
    elif normal_gate["status"] == "Normal model finding":
        sentences.append(
            "The normal gate favors a study with no supported "
            "abnormal finding."
        )
    else:
        sentences.append(
            "No elevated finding outputs were identified, but "
            "the normal gate did not confidently classify the "
            "study as normal."
        )

    if indeterminate:
        sentences.append(
            "Indeterminate outputs are present for "
            + ", ".join(
                finding["name"] for finding in indeterminate[:5]
            )
            + "."
        )

    if negative:
        sentences.append(
            "Lower model outputs are present for "
            + ", ".join(finding["name"] for finding in negative[:4])
            + "."
        )

    impression = [
        f"Possible {finding['name'].lower()} "
        f"(model score {finding['display_score']})."
        for finding in positive[:5]
    ]
    impression += [
        f"Indeterminate {finding['name'].lower()} output "
        f"({finding['display_score']})."
        for finding in indeterminate[:3]
    ]

    if not impression:
        if normal_gate["status"] == "Normal model finding":
            impression = [
                "Normal gate favors no supported abnormal "
                "chest finding."
            ]
        else:
            impression = [
                "No positive finding output; the normal gate "
                f"is {normal_gate['status'].lower()}."
            ]

    confidence = (
        "Lower"
        if len(indeterminate) >= 4
        else "Moderate"
        if indeterminate
        else "Higher"
    )

    return " ".join(sentences), impression, confidence


def build_summary_finding(
    findings: list[dict[str, Any]],
    normal_gate: dict[str, Any],
) -> dict[str, Any]:

    concerning = [
        finding
        for finding in findings
        if finding["status"] != "Negative model finding"
    ]

    if concerning:
        selected = max(
            concerning,
            key=lambda item: item["score"],
        )

        return {
            **selected,
            "is_normal_summary": False,
            "summary_label": selected["name"],
            "summary_status": selected["status"],
        }

    if normal_gate["status"] == "Normal model finding":
        return {
            "name": "No concerning findings detected",
            "score": normal_gate["score"],
            "display_score": normal_gate["display_score"],
            "status": "Negative model finding",
            "threshold": normal_gate["threshold"],
            "display_threshold": normal_gate["display_threshold"],
            "is_normal_summary": True,
            "summary_label": "No concerning findings detected",
            "summary_status": "Negative model finding",
        }

    selected = max(
        findings,
        key=lambda item: item["score"],
    )

    return {
        **selected,
        "is_normal_summary": False,
        "summary_label": selected["name"],
        "summary_status": normal_gate["status"],
    }


def findings_for_display(
    findings: list[dict[str, Any]],
    _summary_finding: dict[str, Any],
) -> list[dict[str, Any]]:

    return findings


def choose_gradcam_finding(
    findings: list[dict[str, Any]],
    summary_finding: dict[str, Any],
) -> dict[str, Any]:

    if (
        not summary_finding["is_normal_summary"]
        and summary_finding["name"] in FINDING_LABELS
    ):
        return summary_finding

    return max(
        findings,
        key=lambda item: item["score"],
    )


def create_gradcam_overlay(
    image_array: np.ndarray,
    input_tensor: torch.Tensor,
    label_index: int,
) -> np.ndarray:

    with GradCAM(
        model=gradcam_model,
        target_layers=[model.features.denseblock4],
    ) as cam:

        heatmap = cam(
            input_tensor=input_tensor,
            targets=[
                MultiLabelOutputTarget(label_index)
            ],
        )[0]

    return show_cam_on_image(
        image_array,
        heatmap,
        use_rgb=True,
    )


def encode_png_base64(image_array: np.ndarray) -> str:
    image = Image.fromarray(image_array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def pil_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def array_to_png_bytes(image_array: np.ndarray) -> bytes:
    return pil_to_png_bytes(Image.fromarray(image_array))


def status_text_color(status: str) -> colors.Color:
    if status == "Positive model finding":
        return colors.HexColor("#BE123C")
    if status == "Indeterminate":
        return colors.HexColor("#B45309")
    return colors.HexColor("#047857")


def draw_page_header_footer(canvas, document) -> None:
    canvas.saveState()

    width, height = letter

    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.setLineWidth(0.6)
    canvas.line(
        document.leftMargin,
        height - 0.58 * inch,
        width - document.rightMargin,
        height - 0.58 * inch,
    )

    canvas.setFont("Helvetica-Bold", 9)
    canvas.setFillColor(colors.HexColor("#0F172A"))
    canvas.drawString(
        document.leftMargin,
        height - 0.42 * inch,
        "MEDVISION - CHEST SPECIALIST",
    )

    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawRightString(
        width - document.rightMargin,
        height - 0.42 * inch,
        f"Page {document.page}",
    )

    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.line(
        document.leftMargin,
        0.55 * inch,
        width - document.rightMargin,
        0.55 * inch,
    )

    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(
        document.leftMargin,
        0.36 * inch,
        "Educational research use only - not intended for medical diagnosis.",
    )

    canvas.restoreState()


def create_pdf_report(
    *,
    filename: str,
    original_image: Image.Image,
    gradcam_overlay: np.ndarray,
    gradcam_finding: str,
    findings: list[dict[str, Any]],
    findings_text: str,
    impression: list[str],
    overall_confidence: str,
) -> bytes:
    output = io.BytesIO()

    document = SimpleDocTemplate(
        output,
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.78 * inch,
        bottomMargin=0.72 * inch,
        title="MedVision Chest Specialist Report",
        author="MedVision",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#0F172A"),
        alignment=TA_LEFT,
        spaceAfter=6,
    )

    eyebrow_style = ParagraphStyle(
        "Eyebrow",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0E7490"),
        spaceAfter=5,
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=4,
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#334155"),
        spaceAfter=7,
    )

    small_style = ParagraphStyle(
        "Small",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748B"),
    )

    centered_small = ParagraphStyle(
        "CenteredSmall",
        parent=small_style,
        alignment=TA_CENTER,
    )

    story: list[Any] = []

    story.append(Paragraph("AI-GENERATED RESEARCH REPORT", eyebrow_style))
    story.append(Paragraph("Chest X-ray Analysis", title_style))
    story.append(
        Paragraph(
            "Dual-head image analysis with a normal gate and eight "
            "supported chest findings.",
            body_style,
        )
    )
    story.append(Spacer(1, 0.08 * inch))

    generated_at = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )

    metadata = [
        ["Study", "Chest X-ray"],
        ["Uploaded file", filename or "Uploaded image"],
        ["Model", "DenseNet-121 dual-head Chest V2"],
        ["Model categories", str(len(FINDING_LABELS))],
        ["Validation combined AUC", f"{VALIDATION_COMBINED_AUC:.4f}"],
        ["Report generated", generated_at],
        ["Overall confidence", overall_confidence],
    ]

    metadata_table = Table(
        metadata,
        colWidths=[1.55 * inch, 5.1 * inch],
        hAlign="LEFT",
    )
    metadata_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#475569")),
                ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#0F172A")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(metadata_table)
    story.append(Spacer(1, 0.2 * inch))

    original_bytes = io.BytesIO(pil_to_png_bytes(original_image))
    gradcam_bytes = io.BytesIO(array_to_png_bytes(gradcam_overlay))

    original_rl_image = ReportLabImage(
        original_bytes,
        width=2.75 * inch,
        height=2.75 * inch,
    )
    gradcam_rl_image = ReportLabImage(
        gradcam_bytes,
        width=2.75 * inch,
        height=2.75 * inch,
    )

    image_table = Table(
        [
            [
                Paragraph("<b>Original image</b>", centered_small),
                Paragraph(
                    f"<b>Grad-CAM - {gradcam_finding}</b>",
                    centered_small,
                ),
            ],
            [original_rl_image, gradcam_rl_image],
        ],
        colWidths=[3.2 * inch, 3.2 * inch],
        hAlign="CENTER",
    )
    image_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F8FAFC")),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(KeepTogether([image_table]))
    story.append(Spacer(1, 0.18 * inch))
    story.append(
        Paragraph(
            "Grad-CAM highlights image regions that influenced the selected "
            "model output. It is not lesion segmentation.",
            small_style,
        )
    )

    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph("Model findings", heading_style))

    finding_rows: list[list[Any]] = [
        [
            Paragraph("<b>Category</b>", small_style),
            Paragraph("<b>Model output</b>", small_style),
            Paragraph("<b>Status</b>", small_style),
            Paragraph("<b>Threshold</b>", small_style),
        ]
    ]

    for finding in findings:
        status_style = ParagraphStyle(
            f"status-{finding['name']}",
            parent=small_style,
            textColor=status_text_color(finding["status"]),
            fontName="Helvetica-Bold",
        )

        finding_rows.append(
            [
                Paragraph(finding["name"], small_style),
                Paragraph(
                    f"<b>{finding['display_score']}</b>",
                    small_style,
                ),
                Paragraph(finding["status"], status_style),
                Paragraph(finding["display_threshold"], small_style),
            ]
        )

    findings_table = Table(
        finding_rows,
        colWidths=[2.28 * inch, 1.12 * inch, 2.0 * inch, 1.0 * inch],
        repeatRows=1,
        hAlign="LEFT",
    )
    findings_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#F8FAFC")],
                ),
            ]
        )
    )
    story.append(findings_table)

    story.append(PageBreak())
    story.append(Paragraph("Findings", heading_style))
    story.append(Paragraph(findings_text, body_style))

    story.append(Spacer(1, 0.08 * inch))
    story.append(Paragraph("Impression", heading_style))

    impression_rows = []
    for index, item in enumerate(impression, start=1):
        impression_rows.append(
            [
                Paragraph(f"<b>{index}</b>", centered_small),
                Paragraph(item, body_style),
            ]
        )

    impression_table = Table(
        impression_rows,
        colWidths=[0.38 * inch, 6.05 * inch],
        hAlign="LEFT",
    )
    impression_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#CFFAFE")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#155E75")),
                ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(impression_table)

    story.append(Spacer(1, 0.22 * inch))
    story.append(Paragraph("How to interpret this report", heading_style))
    story.append(
        Paragraph(
            "The displayed percentages are independent model outputs for "
            "imaging observations. They are not guaranteed disease "
            "probabilities, do not sum to 100%, and are not diagnoses. "
            "Positive, indeterminate, and negative labels are assigned using "
            "validation-derived thresholds. Clinical history, additional "
            "views, prior studies, and clinician interpretation are not "
            "incorporated.",
            body_style,
        )
    )

    story.append(Spacer(1, 0.08 * inch))
    story.append(Paragraph("Limitations", heading_style))
    story.append(
        Paragraph(
            "This system was developed for educational research. Performance "
            "may differ across scanners, institutions, patient populations, "
            "image quality, positioning, and unsupported study types. The "
            "validation set used for the displayed summary metric is small. "
            "The report must not be used to make clinical decisions.",
            body_style,
        )
    )

    document.build(
        story,
        onFirstPage=draw_page_header_footer,
        onLaterPages=draw_page_header_footer,
    )

    output.seek(0)
    return output.getvalue()


def analyze_uploaded_image(
    image: Image.Image,
) -> dict[str, Any]:

    print("CHEST V2: preparing image", flush=True)

    image_array, input_tensor, resized_image = prepare_image(image)

    print(
        "CHEST V2: running dual-head model",
        flush=True,
    )

    (
        normal_probability,
        finding_probabilities,
    ) = get_probabilities(input_tensor)

    normal_gate = classify_normal_gate(normal_probability)

    raw_findings = build_findings(finding_probabilities)

    summary_finding = build_summary_finding(
        raw_findings,
        normal_gate,
    )

    display_findings = findings_for_display(
        raw_findings,
        summary_finding,
    )

    (
        findings_text,
        impression,
        overall_confidence,
    ) = build_report(
        raw_findings,
        normal_gate,
    )

    gradcam_finding = choose_gradcam_finding(
        raw_findings,
        summary_finding,
    )

    print(
        "CHEST V2: generating Grad-CAM for "
        f"{gradcam_finding['name']}",
        flush=True,
    )

    gradcam_index = FINDING_LABELS.index(
        gradcam_finding["name"]
    )

    overlay = create_gradcam_overlay(
        image_array,
        input_tensor,
        gradcam_index,
    )

    print(
        "CHEST V2: analysis complete",
        flush=True,
    )

    return {
        "resized_image": resized_image,
        "normal_gate": normal_gate,
        "findings": display_findings,
        "raw_findings": raw_findings,
        "summary_finding": summary_finding,
        "findings_text": findings_text,
        "impression": impression,
        "overall_confidence": overall_confidence,
        "gradcam_finding": gradcam_finding,
        "overlay": overlay,
    }


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "MedVision API is running"}


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model": "DenseNet-121 dual-head Chest V2",
        "specialist": "Chest",
        "labels": len(FINDING_LABELS),
        "validation_combined_auc": VALIDATION_COMBINED_AUC,
        "pdf_reports": True,
    }


@app.post("/api/chest/predict")
async def predict_chest(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    if file.content_type not in {"image/png", "image/jpeg"}:
        raise HTTPException(
            status_code=400,
            detail="Please upload a PNG or JPEG chest X-ray image.",
        )

    try:
        image = Image.open(io.BytesIO(await file.read()))
        validation = CHEST_VALIDATOR.evaluate(image)
        if not validation.accepted:
            raise HTTPException(
                status_code=422,
                detail=(
                    "This image does not appear to be a supported chest X-ray. "
                    f"Validator score: {validation.display_score}. "
                    "Please upload a frontal or lateral chest radiograph."
                ),
            )
        analysis = analyze_uploaded_image(image)

        gradcam_finding = analysis["gradcam_finding"]
        overlay_base64 = encode_png_base64(analysis["overlay"])

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
                "type": "Chest X-ray",
                "specialist": "Chest Specialist",
                "model": "DenseNet-121 dual-head Chest V2",
                "model_labels": len(FINDING_LABELS),
                "validation_combined_auc": VALIDATION_COMBINED_AUC,
            },
            "normal_gate": analysis["normal_gate"],
            "findings": analysis["findings"],
            "summary_finding": analysis["summary_finding"],
            "findings_text": analysis["findings_text"],
            "impression": analysis["impression"],
            "overall_confidence": analysis["overall_confidence"],
            "gradcam": {
                "finding": gradcam_finding["name"],
                "image_base64": overlay_base64,
                "explanation": (
                    "The heatmap shows image regions that influenced this "
                    "model output. It is not lesion segmentation."
                ),
            },
            "disclaimer": (
                "Educational research use only. "
                "Not intended for medical diagnosis."
            ),
            "classification": gradcam_finding["name"],
            "probability": gradcam_finding["score"],
            "display_probability": gradcam_finding["display_score"],
            "threshold": gradcam_finding["threshold"],
            "gradcam_base64": overlay_base64,
        }

    except HTTPException:
        raise
    except ValidatorUnavailableError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "The chest input validator is not trained or available. "
                "The medical model was not run."
            ),
        ) from error
    except UnidentifiedImageError as error:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        ) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/api/chest/report")
async def download_chest_report(
    file: UploadFile = File(...),
) -> StreamingResponse:
    if file.content_type not in {"image/png", "image/jpeg"}:
        raise HTTPException(
            status_code=400,
            detail="Please upload a PNG or JPEG chest X-ray image.",
        )

    try:
        image = Image.open(io.BytesIO(await file.read()))
        validation = CHEST_VALIDATOR.evaluate(image)
        if not validation.accepted:
            raise HTTPException(
                status_code=422,
                detail=(
                    "This image does not appear to be a supported chest X-ray. "
                    f"Validator score: {validation.display_score}. "
                    "Please upload a frontal or lateral chest radiograph."
                ),
            )
        analysis = analyze_uploaded_image(image)

        pdf_bytes = create_pdf_report(
            filename=file.filename or "uploaded-image",
            original_image=analysis["resized_image"],
            gradcam_overlay=analysis["overlay"],
            gradcam_finding=analysis["gradcam_finding"]["name"],
            findings=analysis["findings"],
            findings_text=analysis["findings_text"],
            impression=analysis["impression"],
            overall_confidence=analysis["overall_confidence"],
        )

        safe_stem = (
            Path(file.filename or "chest-xray").stem
            .replace(" ", "-")
            .replace("_", "-")
        )
        report_filename = f"medvision-{safe_stem}-report.pdf"

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{report_filename}"'
                )
            },
        )

    except HTTPException:
        raise
    except ValidatorUnavailableError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "The chest input validator is not trained or available. "
                "The medical model was not run."
            ),
        ) from error
    except UnidentifiedImageError as error:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        ) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/predict")
async def predict_legacy(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    return await predict_chest(file)