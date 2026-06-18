"""Preprocessing and label mapping tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brain_tumor.config import load_config
from brain_tumor.data import collect_labeled_paths
from brain_tumor.inference import load_class_names, preprocess_rgb_array, raw_label_to_display


def test_preprocess_keeps_zero_to_255_range():
    rgb = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    arr, original = preprocess_rgb_array(rgb, (224, 224))
    assert arr.shape == (1, 224, 224, 3)
    assert arr.dtype == np.float32
    assert arr.min() >= 0.0
    assert arr.max() <= 255.0
    assert original.dtype == np.uint8


def test_preprocess_does_not_divide_by_255():
    rgb = np.full((100, 100, 3), 200, dtype=np.uint8)
    arr, _ = preprocess_rgb_array(rgb, (224, 224))
    assert arr.mean() > 1.0


def test_label_mapping_order():
    labels_path = ROOT / "models" / "class_labels.json"
    if not labels_path.exists():
        pytest.skip("class_labels.json not generated yet")
    names = load_class_names(str(labels_path))
    with open(labels_path, encoding="utf-8") as f:
        mapping = json.load(f)
    assert names[mapping["no"]] == "no"
    assert names[mapping["yes"]] == "yes"


def test_raw_label_to_display():
    assert raw_label_to_display("yes") == "Tumor"
    assert raw_label_to_display("no") == "No Tumor"


def test_dataset_layout_if_present():
    data_dir = ROOT / "dataset"
    if not data_dir.exists():
        pytest.skip("dataset/ not present")
    paths, labels, class_indices = collect_labeled_paths(str(data_dir))
    assert class_indices == {"no": 0, "yes": 1}
    assert len(paths) == len(labels)
    assert len(paths) > 0


def test_config_loads():
    cfg = load_config()
    assert cfg["img_size"] == [224, 224]
    assert cfg["num_classes"] == 2
    assert "paths" in cfg
