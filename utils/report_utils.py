"""Utilities for clinical PDF report generation."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# Display labels for validation metrics in PDF / UI
METRIC_DISPLAY_KEYS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "precision_weighted": "Precision",
    "recall": "Recall",
    "recall_weighted": "Recall",
    "f1_score": "F1 Score",
    "f1_weighted": "F1 Score",
}

METRIC_PRIORITY = ("accuracy", "precision_weighted", "recall_weighted", "f1_weighted")


def normalize_patient_name(name: str) -> str:
    """Collapse whitespace and strip patient name input."""
    return " ".join(name.strip().split())


def patient_display_name(name: str) -> str:
    """Return a display-safe patient name for reports."""
    cleaned = normalize_patient_name(name)
    return cleaned if cleaned else "Not Provided"


def generate_patient_id() -> str:
    """Return a short uppercase patient identifier."""
    return str(uuid.uuid4()).upper()


def format_timestamp(dt: datetime | None = None) -> str:
    dt = dt or datetime.now()
    return dt.strftime("%B %d, %Y at %H:%M:%S")


def report_filename(dt: datetime | None = None) -> str:
    dt = dt or datetime.now()
    return f"report_{dt.strftime('%Y%m%d_%H%M%S')}.pdf"


def assess_risk(confidence: float) -> tuple[str, str]:
    """
    Map model confidence to clinical risk tier.

    Returns (label, description).
    """
    pct = confidence * 100
    if pct > 90:
        return "High Confidence", "The model prediction exceeds 90% confidence."
    if pct >= 70:
        return "Moderate Confidence", "The model prediction is between 70% and 90% confidence."
    return "Low Confidence", "The model prediction is below 70% confidence; manual review is advised."


def generate_clinical_summary(prediction: str, confidence: float) -> str:
    """Dynamic narrative for the clinical summary section."""
    pct = confidence * 100
    if prediction == "Tumor":
        return (
            f"The uploaded MRI scan was analyzed using an EfficientNetB0-based deep learning "
            f"model. The model identified imaging features consistent with a tumor with a "
            f"confidence score of {pct:.1f}%."
        )
    return (
        "The uploaded MRI scan was analyzed using an EfficientNetB0-based deep learning model. "
        "No significant tumor-related imaging features were detected in the submitted slice."
    )


def scan_resolution_label(height: int, width: int) -> str:
    return f"{width} x {height} pixels"


def load_validation_metrics(metrics_path: str | Path) -> dict[str, float]:
    """
    Load results/metrics.json and normalize to the four portfolio metrics
    shown in the clinical report.
    """
    path = Path(metrics_path)
    if not path.exists():
        return {}

    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)

    aliases = {
        "accuracy": ["accuracy"],
        "precision": ["precision_weighted", "precision"],
        "recall": ["recall_weighted", "recall"],
        "f1_score": ["f1_weighted", "f1_score"],
    }
    out: dict[str, float] = {}
    for display_key, candidates in aliases.items():
        for key in candidates:
            if key in raw:
                out[display_key] = float(raw[key])
                break
    return out


def metrics_table_rows(metrics: dict[str, float]) -> list[tuple[str, str]]:
    """Format metrics as (label, percentage string) rows for PDF tables."""
    labels = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1 Score",
    }
    rows = []
    for key in ("accuracy", "precision", "recall", "f1_score"):
        if key in metrics:
            rows.append((labels[key], f"{metrics[key] * 100:.1f}%"))
    return rows
