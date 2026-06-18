"""Smoke tests for inference (skipped when model weights absent)."""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brain_tumor.config import load_config, resolve_path
from brain_tumor.inference import predict


@pytest.fixture
def cfg():
    return load_config()


def test_predict_on_sample_image(cfg):
    model_path = resolve_path(cfg, "model_path")
    if not Path(model_path).exists():
        pytest.skip("Trained model not found")

    dataset = ROOT / "dataset"
    if not dataset.exists():
        pytest.skip("dataset/ not found")

    candidates = glob.glob(str(dataset / "yes" / "*")) + glob.glob(str(dataset / "no" / "*"))
    if not candidates:
        pytest.skip("No sample images")

    result = predict(candidates[0], save_gradcam=False, cfg=cfg)
    assert result["prediction"] in ("Tumor", "No Tumor")
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["all_probs"]) == 2
    assert abs(result["all_probs"].sum() - 1.0) < 0.01
