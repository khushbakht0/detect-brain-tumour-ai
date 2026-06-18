#!/usr/bin/env python3
"""
Stratified k-fold cross-validation for the brain tumor classifier.

Trains a fresh model per fold (fewer epochs by default for runtime).
Full training metrics are written to results/cv_metrics.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brain_tumor.config import img_size_tuple, load_config, resolve_path
from brain_tumor.data import compute_class_weights
from brain_tumor.evaluation import medical_metrics, run_cross_validation
from brain_tumor.inference import load_and_preprocess
from brain_tumor.model import build_model


def load_batch(paths, labels, img_size):
    xs, ys = [], []
    for path, label in zip(paths, labels):
        img_array, _ = load_and_preprocess(path, img_size)
        xs.append(img_array[0])
        ys.append(label)
    return np.stack(xs), to_categorical(ys, num_classes=2)


def make_fold_trainer(cfg, epochs_per_fold: int):
    img_size = img_size_tuple(cfg)
    seed = cfg["seed"]

    def train_fold(fold, train_paths, train_labels, val_paths, val_labels, class_indices):
        del class_indices
        tf.random.set_seed(seed + fold)
        np.random.seed(seed + fold)

        x_train, y_train = load_batch(train_paths, train_labels, img_size)
        x_val, y_val = load_batch(val_paths, val_labels, img_size)

        class_weights = compute_class_weights_from_labels(train_labels)

        model, _ = build_model(img_size, cfg["num_classes"], freeze_base=True)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=cfg["training"]["learning_rate_head"]),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs_per_fold,
            batch_size=cfg["batch_size"],
            class_weight=class_weights,
            verbose=0,
        )

        y_prob = model.predict(x_val, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        y_true = np.array(val_labels)
        return y_true, y_pred, y_prob

    return train_fold


def compute_class_weights_from_labels(labels):
    labels = np.array(labels)
    classes = np.unique(labels)
    from sklearn.utils.class_weight import compute_class_weight

    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def parse_args():
    parser = argparse.ArgumentParser(description="Stratified k-fold cross-validation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--epochs-per-fold", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    n_splits = args.folds or cfg["evaluation"]["cv_folds"]
    data_dir = resolve_path(cfg, "data_dir")

    print(f"[CV] {n_splits}-fold stratified cross-validation on {data_dir}")
    trainer = make_fold_trainer(cfg, args.epochs_per_fold)
    run_cross_validation(cfg, trainer, data_dir, n_splits=n_splits)


if __name__ == "__main__":
    main()
