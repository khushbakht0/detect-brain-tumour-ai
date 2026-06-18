"""Load and validate project configuration from YAML."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "default.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = ROOT / path
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(ROOT)
    cfg["_config_path"] = str(path)
    return cfg


def resolve_path(cfg: dict[str, Any], key: str) -> str:
    """Resolve a path under paths.* relative to project root."""
    value = cfg["paths"][key]
    if os.path.isabs(value):
        return value
    return str(ROOT / value)


def img_size_tuple(cfg: dict[str, Any]) -> tuple[int, int]:
    w, h = cfg["img_size"]
    return int(h), int(w)
