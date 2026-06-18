"""Tests for clinical report utilities and PDF generation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from report_generator import ReportPayload, generate_clinical_report
from utils.report_utils import (
    assess_risk,
    generate_clinical_summary,
    load_validation_metrics,
    report_filename,
)


def test_assess_risk_tiers():
    assert assess_risk(0.95)[0] == "High Confidence"
    assert assess_risk(0.80)[0] == "Moderate Confidence"
    assert assess_risk(0.60)[0] == "Low Confidence"


def test_clinical_summary_tumor():
    text = generate_clinical_summary("Tumor", 0.924)
    assert "tumor" in text.lower()
    assert "92.4" in text


def test_clinical_summary_no_tumor():
    text = generate_clinical_summary("No Tumor", 0.88)
    assert "no significant tumor" in text.lower()


def test_load_validation_metrics():
    metrics_path = ROOT / "results" / "metrics.json"
    if not metrics_path.exists():
        pytest.skip("metrics.json missing")
    m = load_validation_metrics(metrics_path)
    assert "accuracy" in m
    assert "precision" in m


def test_generate_pdf_bytes(tmp_path):
    payload = ReportPayload(
        scan_filename="test_scan.jpg",
        patient_name="Jane Doe",
        prediction="Tumor",
        confidence=0.92,
        original_img=np.zeros((224, 224, 3), dtype=np.uint8),
        heatmap_rgb=np.full((224, 224, 3), 128, dtype=np.uint8),
        overlay=np.full((224, 224, 3), 64, dtype=np.uint8),
        reports_dir=str(tmp_path),
        metrics_path=str(ROOT / "results" / "metrics.json"),
    )
    pdf_bytes, filepath = generate_clinical_report(payload, save=True)
    assert pdf_bytes[:4] == b"%PDF"
    assert Path(filepath).exists()
    assert filepath.endswith(".pdf")
    assert report_filename()[:7] == "report_"
