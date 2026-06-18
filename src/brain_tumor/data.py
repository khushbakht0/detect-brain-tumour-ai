"""Data loading, augmentation generators, and dataset utilities."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from brain_tumor.config import img_size_tuple


def build_generators(
    data_dir: str,
    val_split: float,
    batch_size: int,
    seed: int,
    img_size: tuple[int, int],
    augmentation: dict | None = None,
):
    """
    Train/val generators. Pixel values stay in [0, 255]; EfficientNetB0
    rescales internally. Do NOT divide by 255 (causes collapsed predictions).
    """
    aug = augmentation or {}
    train_datagen = ImageDataGenerator(
        validation_split=val_split,
        rotation_range=aug.get("rotation_range", 20),
        zoom_range=aug.get("zoom_range", 0.15),
        width_shift_range=aug.get("width_shift_range", 0.1),
        height_shift_range=aug.get("height_shift_range", 0.1),
        horizontal_flip=aug.get("horizontal_flip", True),
        vertical_flip=aug.get("vertical_flip", False),
        brightness_range=aug.get("brightness_range", [0.85, 1.15]),
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(validation_split=val_split)

    common = dict(
        directory=data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        seed=seed,
    )
    train_gen = train_datagen.flow_from_directory(subset="training", shuffle=True, **common)
    val_gen = val_datagen.flow_from_directory(subset="validation", shuffle=False, **common)
    return train_gen, val_gen


def compute_class_weights(train_gen) -> dict[int, float]:
    classes = np.unique(train_gen.classes)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=train_gen.classes
    )
    return {int(c): float(w) for c, w in zip(classes, weights)}


def collect_labeled_paths(data_dir: str) -> tuple[list[str], list[int], dict[str, int]]:
    """
    Walk yes/ and no/ folders and return paths, integer labels, class_indices.
    Matches Keras flow_from_directory alphabetical ordering.
    """
    data_path = Path(data_dir)
    class_names = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    class_indices = {name: idx for idx, name in enumerate(class_names)}

    paths, labels = [], []
    for name in class_names:
        folder = data_path / name
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            for fp in sorted(folder.glob(ext)):
                paths.append(str(fp))
                labels.append(class_indices[name])
    return paths, labels, class_indices


def count_class_distribution(data_dir: str) -> dict[str, int]:
    paths, labels, class_indices = collect_labeled_paths(data_dir)
    inv = {v: k for k, v in class_indices.items()}
    counts = {name: 0 for name in class_indices}
    for label in labels:
        counts[inv[label]] += 1
    return counts
