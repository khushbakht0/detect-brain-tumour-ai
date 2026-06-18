#!/usr/bin/env python3
"""
Download or prepare the brain tumor MRI dataset.

Option A — Kaggle (recommended):
  1. Install Kaggle CLI: pip install kaggle
  2. Place kaggle.json in ~/.kaggle/ (or %USERPROFILE%\\.kaggle\\ on Windows)
  3. Run: python scripts/download_data.py --kaggle

Option B — Manual:
  Download from https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
  Extract so that dataset/yes/ and dataset/no/ exist, then run:
  python scripts/download_data.py --verify
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "dataset"
KAGGLE_DATASET = "masoudnickparvar/brain-tumor-mri-dataset"


def verify_dataset(data_dir: Path) -> bool:
    yes_dir = data_dir / "yes"
    no_dir = data_dir / "no"
    if not yes_dir.is_dir() or not no_dir.is_dir():
        print(f"[ERROR] Expected {yes_dir} and {no_dir} to exist.")
        return False
    n_yes = sum(1 for _ in yes_dir.iterdir() if _.is_file())
    n_no = sum(1 for _ in no_dir.iterdir() if _.is_file())
    print(f"[OK] Dataset verified: yes={n_yes}, no={n_no}, total={n_yes + n_no}")
    return n_yes > 0 and n_no > 0


def normalize_kaggle_layout(data_dir: Path) -> None:
    """Some Kaggle archives nest folders; flatten to dataset/yes and dataset/no."""
    for sub in data_dir.rglob("*"):
        if sub.is_dir() and sub.name in ("yes", "no") and sub.parent != data_dir:
            target = data_dir / sub.name
            target.mkdir(parents=True, exist_ok=True)
            for fp in sub.iterdir():
                if fp.is_file():
                    dest = target / fp.name
                    if not dest.exists():
                        shutil.move(str(fp), str(dest))
            print(f"[OK] Merged {sub} -> {target}")


def download_kaggle(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "brain-tumor-mri-dataset.zip"
    print(f"[Download] Fetching {KAGGLE_DATASET} via Kaggle API...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "download",
            "-d",
            KAGGLE_DATASET,
            "-p",
            str(data_dir),
            "--unzip",
        ],
        check=True,
    )
    if archive.exists():
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(data_dir)
        archive.unlink(missing_ok=True)
    normalize_kaggle_layout(data_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare brain tumor MRI dataset")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--kaggle", action="store_true", help="Download via Kaggle API")
    parser.add_argument("--verify", action="store_true", help="Verify existing layout only")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    if args.kaggle:
        download_kaggle(data_dir)
    if not verify_dataset(data_dir):
        print(
            "\nDataset not ready. See scripts/download_data.py docstring for setup steps."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
