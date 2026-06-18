"""Training orchestration."""

from __future__ import annotations

import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, optimizers

from brain_tumor.config import img_size_tuple, load_config, resolve_path
from brain_tumor.data import build_generators, compute_class_weights
from brain_tumor.evaluation import evaluate_model, save_confusion_matrix_plot, save_training_curves
from brain_tumor.model import build_model, unfreeze_top_layers


def set_seed(seed: int) -> None:
    tf.random.set_seed(seed)
    np.random.seed(seed)


def build_training_callbacks(cfg: dict, model_path: str, phase: int = 1):
    training = cfg["training"]
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=training["early_stopping_patience_phase1"]
            if phase == 1
            else training["early_stopping_patience_phase2"],
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=training["reduce_lr_factor"],
            patience=training["reduce_lr_patience"],
            min_lr=training["min_lr"],
            verbose=1,
        ),
        callbacks.TensorBoard(log_dir=resolve_path(cfg, "logs_dir"), histogram_freq=1),
    ]


def train(cfg: dict | None = None, overrides: dict | None = None) -> dict:
    cfg = load_config() if cfg is None else cfg
    if overrides:
        for key, value in overrides.items():
            if key in cfg and isinstance(cfg[key], dict) and isinstance(value, dict):
                cfg[key].update(value)
            elif key == "paths" and isinstance(value, dict):
                cfg["paths"].update(value)
            else:
                cfg[key] = value

    set_seed(cfg["seed"])
    data_dir = resolve_path(cfg, "data_dir")
    model_path = resolve_path(cfg, "model_path")
    labels_path = resolve_path(cfg, "labels_path")
    results_dir = resolve_path(cfg, "results_dir")
    img_size = img_size_tuple(cfg)

    print("\n" + "=" * 55)
    print("   Brain Tumor Detection — Training Pipeline")
    print("=" * 55)
    print(f"  Data dir    : {data_dir}")
    print(f"  Image size  : {img_size}")
    print(f"  Epochs      : {cfg['epochs']}")
    print(f"  Batch size  : {cfg['batch_size']}")
    print(f"  Fine-tuning : {cfg['fine_tune']}")
    print("=" * 55 + "\n")

    train_gen, val_gen = build_generators(
        data_dir,
        cfg["val_split"],
        cfg["batch_size"],
        cfg["seed"],
        img_size,
        cfg.get("augmentation"),
    )
    class_names = list(train_gen.class_indices.keys())
    class_weights = compute_class_weights(train_gen)
    print(f"[Data] Classes detected: {train_gen.class_indices}")
    print(f"[Data] Val class indices:   {val_gen.class_indices}")
    print(f"[Data] Class weights:       {class_weights}")
    print(f"[Data] Train samples: {train_gen.samples}, Val samples: {val_gen.samples}")
    assert train_gen.class_indices == val_gen.class_indices

    os.makedirs(os.path.dirname(labels_path) or ".", exist_ok=True)
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(train_gen.class_indices, f, indent=2)

    model, base = build_model(img_size, cfg["num_classes"], freeze_base=True)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=cfg["training"]["learning_rate_head"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    phase1_epochs = max(1, cfg["epochs"] // 2)
    print(f"\n[Phase 1] Training classification head ({phase1_epochs} epochs)...")
    hist1 = model.fit(
        train_gen,
        epochs=phase1_epochs,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=build_training_callbacks(cfg, model_path, phase=1),
        verbose=1,
    )

    if cfg["fine_tune"]:
        unfreeze_top_layers(model, base, cfg["unfreeze_layers"])
        model.compile(
            optimizer=optimizers.Adam(
                learning_rate=cfg["training"]["learning_rate_finetune"]
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        phase2_epochs = cfg["epochs"] - phase1_epochs
        print(f"\n[Phase 2] Fine-tuning top base layers ({phase2_epochs} epochs)...")
        hist2 = model.fit(
            train_gen,
            epochs=phase2_epochs,
            validation_data=val_gen,
            class_weight=class_weights,
            callbacks=build_training_callbacks(cfg, model_path, phase=2),
            verbose=1,
        )
        for key in hist1.history:
            hist1.history[key].extend(hist2.history.get(key, []))

    print(f"\n[OK] Best model weights -> {model_path}")
    save_training_curves(hist1, os.path.join(results_dir, "accuracy_plot.png"))
    y_true, y_pred, metrics = evaluate_model(
        model,
        val_gen,
        class_names,
        results_dir,
        positive_index=cfg["evaluation"]["positive_class_index"],
    )
    save_confusion_matrix_plot(
        y_true, y_pred, class_names, os.path.join(results_dir, "confusion_matrix.png")
    )
    print("\n[OK] Training complete.")
    return metrics
