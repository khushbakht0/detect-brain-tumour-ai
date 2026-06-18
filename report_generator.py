"""
Clinical-style PDF report generator for Brain Tumor AI predictions.

Uses ReportLab to produce radiology-style analysis reports with embedded
Grad-CAM visualizations and validation metrics.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from utils.report_utils import (
    assess_risk,
    format_timestamp,
    generate_clinical_summary,
    generate_patient_id,
    load_validation_metrics,
    metrics_table_rows,
    report_filename,
    scan_resolution_label,
    patient_display_name,
)

# Brand palette
PRIMARY = colors.HexColor("#2563EB")
SECONDARY = colors.HexColor("#0F172A")
SUCCESS = colors.HexColor("#22C55E")
DANGER = colors.HexColor("#EF4444")
MUTED = colors.HexColor("#64748B")
# Off-white clinical palette
BG_OFFWHITE = "#F5F4F0"
BG_WARM = "#FAF9F6"
BORDER = colors.HexColor("#D4D4CE")
LIGHT_BG = colors.HexColor(BG_WARM)

AI_SYSTEM_VERSION = "1.0.0"
DEFAULT_REPORTS_DIR = "reports"


@dataclass
class ReportPayload:
    """Inputs required to render a clinical PDF report."""

    scan_filename: str
    prediction: str
    confidence: float
    original_img: np.ndarray
    heatmap_rgb: np.ndarray
    overlay: np.ndarray
    patient_id: str = field(default_factory=generate_patient_id)
    patient_name: str = "Not Provided"
    generated_at: datetime = field(default_factory=datetime.now)
    ai_version: str = AI_SYSTEM_VERSION
    metrics_path: str = "results/metrics.json"
    reports_dir: str = DEFAULT_REPORTS_DIR


def _numpy_to_rl_image(arr: np.ndarray, width: float, height: float) -> RLImage:
    pil = PILImage.fromarray(arr.astype(np.uint8))
    buf = BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return RLImage(buf, width=width, height=height)


def _build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Heading1"],
            fontSize=22,
            textColor=SECONDARY,
            spaceAfter=2,
            alignment=TA_LEFT,
            fontName="Helvetica-Bold",
        ),
        "finding": ParagraphStyle(
            "FindingTitle",
            parent=base["Heading1"],
            fontSize=16,
            textColor=PRIMARY,
            spaceBefore=4,
            spaceAfter=10,
            alignment=TA_LEFT,
            fontName="Helvetica-Bold",
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontSize=10,
            textColor=MUTED,
            spaceAfter=2,
        ),
        "section": ParagraphStyle(
            "SectionHead",
            parent=base["Heading2"],
            fontSize=12,
            textColor=PRIMARY,
            spaceBefore=14,
            spaceAfter=8,
            fontName="Helvetica-Bold",
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["Normal"],
            fontSize=10,
            textColor=SECONDARY,
            leading=14,
            alignment=TA_JUSTIFY,
        ),
        "label": ParagraphStyle(
            "Label",
            parent=base["Normal"],
            fontSize=9,
            textColor=MUTED,
            fontName="Helvetica-Bold",
        ),
        "value": ParagraphStyle(
            "Value",
            parent=base["Normal"],
            fontSize=10,
            textColor=SECONDARY,
        ),
        "footer": ParagraphStyle(
            "Footer",
            parent=base["Normal"],
            fontSize=8,
            textColor=MUTED,
            alignment=TA_CENTER,
            leading=11,
        ),
        "disclaimer": ParagraphStyle(
            "Disclaimer",
            parent=base["Normal"],
            fontSize=8,
            textColor=DANGER,
            alignment=TA_JUSTIFY,
            leading=11,
            fontName="Helvetica-Bold",
        ),
    }


def _cell(content, style_key: str = "value") -> Paragraph:
    styles = _build_styles()
    if isinstance(content, Paragraph):
        return content
    return Paragraph(str(content), styles[style_key])


def _info_table(rows: list[tuple[str, Any]], col_widths=None) -> Table:
    """Minimal row layout — hairline separators only, no box fill."""
    styles = _build_styles()
    data = [[Paragraph(k, styles["label"]), _cell(v)] for k, v in rows]
    table = Table(data, colWidths=col_widths or [1.55 * inch, 4.65 * inch])
    table.setStyle(
        TableStyle(
            [
                ("LINEBELOW", (0, 0), (-1, -2), 0.25, BORDER),
                ("LINEBELOW", (0, -1), (-1, -1), 0.5, BORDER),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def _draw_page_frame(canvas, doc) -> None:
    """Thin off-white page with minimal outer border."""
    canvas.saveState()
    canvas.setFillColorRGB(0.961, 0.957, 0.941)  # #F5F4F0
    canvas.rect(0, 0, letter[0], letter[1], fill=1, stroke=0)
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.5)
    inset = 0.4 * inch
    canvas.rect(
        inset,
        inset,
        letter[0] - 2 * inset,
        letter[1] - 2 * inset,
        fill=0,
        stroke=1,
    )
    canvas.restoreState()


def _finding_title(prediction: str) -> str:
    return "Tumor Detected" if prediction == "Tumor" else "No Tumor Detected"


def _header_block(styles: dict, payload: ReportPayload) -> list:
    finding = _finding_title(payload.prediction)
    finding_color = DANGER if payload.prediction == "Tumor" else SUCCESS
    finding_style = ParagraphStyle(
        "Finding",
        parent=styles["finding"],
        textColor=finding_color,
    )
    story = [
        Paragraph("Brain Tumor AI Analysis Report", styles["title"]),
        Paragraph(finding, finding_style),
        Paragraph(f"Generated: {format_timestamp(payload.generated_at)}", styles["subtitle"]),
        Paragraph(f"AI System Version: {payload.ai_version}", styles["subtitle"]),
        Spacer(1, 0.12 * inch),
        HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=14),
    ]
    return story


def generate_clinical_report(
    payload: ReportPayload,
    save: bool = True,
) -> tuple[bytes, str]:
    """
    Build a clinical PDF report.

    Parameters
    ----------
    payload : ReportPayload
    save    : If True, write PDF to reports/report_<timestamp>.pdf

    Returns
    -------
    pdf_bytes : Raw PDF content for Streamlit download
    filepath  : Absolute or relative path where PDF was saved (empty if save=False)
    """
    styles = _build_styles()
    filename = report_filename(payload.generated_at)
    reports_dir = Path(payload.reports_dir)
    filepath = str(reports_dir / filename)

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="Brain Tumor AI Analysis Report",
        author="Brain Tumor AI System",
    )

    h, w = payload.original_img.shape[:2]
    risk_label, risk_desc = assess_risk(payload.confidence)
    summary = generate_clinical_summary(payload.prediction, payload.confidence)
    metrics = load_validation_metrics(payload.metrics_path)

    pred_color = DANGER if payload.prediction == "Tumor" else SUCCESS
    pred_style = ParagraphStyle(
        "PredValue",
        parent=styles["value"],
        textColor=pred_color,
        fontName="Helvetica-Bold",
        fontSize=12,
    )

    story: list[Any] = []
    story.extend(_header_block(styles, payload))

    # Patient section
    story.append(Paragraph("Patient Information", styles["section"]))
    story.append(
        _info_table(
            [
                ("Patient Name", patient_display_name(payload.patient_name)),
                ("Patient ID", payload.patient_id),
                ("Scan Filename", payload.scan_filename),
                ("Scan Resolution", scan_resolution_label(h, w)),
            ]
        )
    )

    # Prediction section
    story.append(Paragraph("Classification Result", styles["section"]))
    story.append(
        _info_table(
            [
                ("Classification", Paragraph(payload.prediction, pred_style)),
                ("Confidence Score", f"{payload.confidence * 100:.1f}%"),
            ]
        )
    )

    # Risk assessment
    story.append(Paragraph("Risk Assessment", styles["section"]))
    risk_style = ParagraphStyle(
        "Risk",
        parent=styles["value"],
        fontName="Helvetica-Bold",
        textColor=PRIMARY if "High" in risk_label else SECONDARY,
    )
    story.append(
        _info_table(
            [
                ("Confidence Tier", Paragraph(risk_label, risk_style)),
                ("Interpretation", risk_desc),
            ]
        )
    )

    # Clinical summary
    story.append(Paragraph("Clinical Summary", styles["section"]))
    story.append(Paragraph(summary, styles["body"]))

    # Explainability images
    story.append(Paragraph("Explainability — Grad-CAM Analysis", styles["section"]))
    img_w = 2.15 * inch
    img_h = 2.15 * inch
    img_row = [
        _numpy_to_rl_image(payload.original_img, img_w, img_h),
        _numpy_to_rl_image(payload.heatmap_rgb, img_w, img_h),
        _numpy_to_rl_image(payload.overlay, img_w, img_h),
    ]
    img_table = Table(
        [img_row],
        colWidths=[img_w + 0.1 * inch] * 3,
    )
    img_table.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (0, 0), 0.5, BORDER),
                ("BOX", (1, 0), (1, 0), 0.5, BORDER),
                ("BOX", (2, 0), (2, 0), 0.5, BORDER),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    caption_data = [
        [
            Paragraph("Original MRI", styles["subtitle"]),
            Paragraph("Grad-CAM Heatmap", styles["subtitle"]),
            Paragraph("Attention Overlay", styles["subtitle"]),
        ]
    ]
    caption_table = Table(caption_data, colWidths=[img_w + 0.1 * inch] * 3)
    caption_table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))

    story.append(img_table)
    story.append(Spacer(1, 0.08 * inch))
    story.append(caption_table)

    # Model performance
    story.append(Paragraph("Model Validation Performance", styles["section"]))
    if metrics:
        perf_rows = metrics_table_rows(metrics)
        perf_data = [
            [Paragraph("Metric", styles["label"]), Paragraph("Validation Score", styles["label"])]
        ] + [[Paragraph(k, styles["value"]), Paragraph(v, styles["value"])] for k, v in perf_rows]
        perf_table = Table(perf_data, colWidths=[2.5 * inch, 2.5 * inch])
        perf_table.setStyle(
            TableStyle(
                [
                    ("LINEBELOW", (0, 0), (-1, 0), 0.5, BORDER),
                    ("LINEBELOW", (0, 1), (-1, -2), 0.25, BORDER),
                    ("LINEBELOW", (0, -1), (-1, -1), 0.5, BORDER),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        story.append(perf_table)
        story.append(Spacer(1, 0.08 * inch))
        story.append(
            Paragraph(
                "Metrics reflect hold-out validation on the training dataset "
                "(see results/metrics.json). Not patient-specific performance.",
                styles["subtitle"],
            )
        )
    else:
        story.append(
            Paragraph(
                "Validation metrics not available. Train the model to populate results/metrics.json.",
                styles["body"],
            )
        )

    # Footer disclaimer
    story.append(Spacer(1, 0.25 * inch))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=10))
    story.append(Paragraph("DISCLAIMER", styles["disclaimer"]))
    story.append(
        Paragraph(
            "This report is generated by an experimental AI system for educational and "
            "research purposes only. It is not intended for medical diagnosis, treatment "
            "planning, or clinical decision-making. Always consult a qualified neuroradiologist.",
            styles["footer"],
        )
    )
    story.append(Spacer(1, 0.1 * inch))
    story.append(
        Paragraph(
            f"Report ID: {payload.patient_id} | Brain Tumor AI v{payload.ai_version}",
            styles["footer"],
        )
    )

    doc.build(story, onFirstPage=_draw_page_frame, onLaterPages=_draw_page_frame)
    pdf_bytes = buffer.getvalue()

    if save:
        os.makedirs(reports_dir, exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(pdf_bytes)

    return pdf_bytes, filepath if save else ""


def build_report_from_prediction(
    result: dict,
    scan_filename: str,
    patient_name: str = "",
    metrics_path: str = "results/metrics.json",
    reports_dir: str = DEFAULT_REPORTS_DIR,
) -> tuple[bytes, str]:
    """Convenience wrapper using the predict() result dict from inference."""
    payload = ReportPayload(
        scan_filename=scan_filename,
        prediction=result["prediction"],
        confidence=float(result["confidence"]),
        original_img=result["original_img"],
        heatmap_rgb=result["heatmap_rgb"],
        overlay=result["overlay"],
        patient_name=patient_name,
        metrics_path=metrics_path,
        reports_dir=reports_dir,
    )
    return generate_clinical_report(payload, save=True)
