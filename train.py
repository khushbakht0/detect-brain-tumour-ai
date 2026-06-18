#!/usr/bin/env python3
"""Train the brain tumor classifier."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from brain_tumor.config import load_config
from brain_tumor.training import train


def parse_args():
    parser = argparse.ArgumentParser(description="Train Brain Tumor Detector")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--val_split", type=float, default=None)
    parser.add_argument("--fine_tune", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    overrides = {}
    if args.data_dir:
        overrides.setdefault("paths", {})["data_dir"] = args.data_dir
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.val_split is not None:
        overrides["val_split"] = args.val_split
    if args.fine_tune:
        overrides["fine_tune"] = True
    train(cfg, overrides=overrides)


if __name__ == "__main__":
    main()
