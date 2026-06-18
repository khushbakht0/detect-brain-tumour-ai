#!/usr/bin/env python3
"""Run inference on a single MRI image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from brain_tumor.inference import predict


def parse_args():
    parser = argparse.ArgumentParser(description="Predict brain tumor from MRI image")
    parser.add_argument("--image", type=str, required=True, help="Path to MRI image")
    parser.add_argument("--no-gradcam", action="store_true", help="Skip Grad-CAM output")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save Grad-CAM figure",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    predict(
        image_path=args.image,
        save_gradcam=not args.no_gradcam,
        gradcam_save_path=args.output,
    )


if __name__ == "__main__":
    main()
