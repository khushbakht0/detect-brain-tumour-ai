"""
train.py — Brain Tumor Detection: Training Pipeline
=====================================================
Uses EfficientNetB0 as a frozen feature extractor with a custom classification
head. Supports full fine-tuning in a second phase. Outputs model weights,
training curves, confusion matrix, and per-class metrics.

Usage:
    python train.py --data_dir dataset/ --epochs 30 --batch_size 32
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless rendering for servers
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.utils.class_weight import compute_class_weight

# ──────────────────────────────────────────────
# Reproducibility seed
# ──────────────────────────────────────────────
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
NUM_CLASSES = 2
MODEL_PATH  = "models/brain_tumor_model.h5"
LABELS_PATH = "models/class_labels.json"
RESULTS_DIR = "results/"


def parse_args():
    parser = argparse.ArgumentParser(description="Train Brain Tumor Detector")
    parser.add_argument("--data_dir",   type=str, default="dataset/",
                        help="Root directory containing yes/ and no/ subdirs")
    parser.add_argument("--epochs",     type=int, default=30,
                        help="Total training epochs (both phases combined)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_split",  type=float, default=0.2,
                        help="Fraction of data used for validation")
    parser.add_argument("--fine_tune",  action="store_true",
                        help="Enable second-phase fine-tuning of base layers")
    return parser.parse_args()


# ──────────────────────────────────────────────
# Data Generators
# ──────────────────────────────────────────────
def build_generators(data_dir, val_split, batch_size):
    """
    Create augmented train generator and validation generator.
    Augmentation is applied only to training data to prevent data leakage.

    NOTE: Do NOT rescale to [0, 1]. Keras 3 EfficientNetB0 includes an internal
    Rescaling(1/255) layer and expects pixel values in [0, 255]. Dividing by
    255 here causes double-rescaling (~black images) and collapsed predictions.
    """
    # Split is deterministic from validation_split + seed (verified: zero overlap).
    # Augmentation on train only — val must not be randomly transformed.
    train_datagen = ImageDataGenerator(
        validation_split=val_split,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.85, 1.15],
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(validation_split=val_split)

    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=SEED,
    )

    val_gen = val_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=SEED,
    )

    return train_gen, val_gen


def compute_class_weights(train_gen):
    """Balanced class weights for imbalanced yes/no folders."""
    classes = np.unique(train_gen.classes)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=train_gen.classes
    )
    return {int(c): float(w) for c, w in zip(classes, weights)}


# ──────────────────────────────────────────────
# Model Architecture
# ──────────────────────────────────────────────
def build_model(num_classes=2, freeze_base=True):
    """
    EfficientNetB0 feature extractor + custom classification head.

    Architecture:
        EfficientNetB0 (ImageNet weights, base frozen initially)
        → GlobalAveragePooling2D
        → BatchNormalization
        → Dense(256, relu) + Dropout(0.4)
        → BatchNormalization
        → Dense(128, relu) + Dropout(0.3)
        → Dense(num_classes, softmax)
    """
    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3)
    )
    base.trainable = not freeze_base  # freeze during phase 1

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    # Let Keras propagate the training flag (required for fine-tuning BN layers).
    # Frozen layers ignore gradients regardless of the training flag.
    x = base(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="BrainTumorDetector")
    return model, base


def unfreeze_top_layers(model, base_model, num_layers_to_unfreeze=30):
    """
    Phase 2: unfreeze the top N layers of EfficientNetB0 for fine-tuning.
    Use a lower learning rate to avoid destroying pre-trained features.
    """
    base_model.trainable = True
    # Freeze everything except the last N layers
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    print(f"\n[Fine-tune] Unfroze top {num_layers_to_unfreeze} base layers.")
    trainable = sum(1 for l in model.layers if l.trainable)
    print(f"[Fine-tune] Trainable layers: {trainable}")


# ──────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────
def build_callbacks(phase=1):
    os.makedirs("models", exist_ok=True)
    cb = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7 if phase == 1 else 5,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.TensorBoard(log_dir="logs/", histogram_freq=1)
    ]
    return cb


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────
def save_training_curves(history, filename="results/accuracy_plot.png"):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Diagnostics — Brain Tumor Detector", fontsize=14,
                 fontweight="bold")

    # --- Accuracy subplot ---
    ax = axes[0]
    ax.plot(history.history["accuracy"],     label="Train Acc",  color="#2196F3", lw=2)
    ax.plot(history.history["val_accuracy"], label="Val Acc",    color="#FF5722", lw=2, ls="--")
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Loss subplot ---
    ax = axes[1]
    ax.plot(history.history["loss"],     label="Train Loss", color="#4CAF50", lw=2)
    ax.plot(history.history["val_loss"], label="Val Loss",   color="#9C27B0", lw=2, ls="--")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Categorical Cross-Entropy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Training curves saved -> {filename}")


def save_confusion_matrix(y_true, y_pred, class_names,
                           filename="results/confusion_matrix.png"):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, linecolor="gray",
        cbar_kws={"shrink": 0.8}, ax=ax
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title("Confusion Matrix — Brain Tumor Detection", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Confusion matrix saved -> {filename}")


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
def evaluate_model(model, val_gen, class_names):
    """
    Run full evaluation: accuracy, precision, recall, F1.
    Returns true labels and predicted labels for further plotting.
    """
    print("\n[Eval] Running inference on validation set...")
    val_gen.reset()
    y_true_batches, y_prob_batches = [], []
    for batch_idx in range(len(val_gen)):
        batch_x, batch_y = val_gen[batch_idx]
        y_prob_batches.append(model.predict(batch_x, verbose=0))
        y_true_batches.append(np.argmax(batch_y, axis=1))

    y_prob = np.vstack(y_prob_batches)
    y_true = np.concatenate(y_true_batches)
    y_pred = np.argmax(y_prob, axis=1)

    print("\n" + "=" * 55)
    print("         CLASSIFICATION REPORT")
    print("=" * 55)
    print(classification_report(y_true, y_pred, target_names=class_names))

    prec  = precision_score(y_true, y_pred, average="weighted")
    rec   = recall_score(   y_true, y_pred, average="weighted")
    f1    = f1_score(        y_true, y_pred, average="weighted")
    acc   = np.mean(y_true == y_pred)

    metrics = {"accuracy": round(acc, 4), "precision": round(prec, 4),
               "recall": round(rec, 4),   "f1_score":  round(f1, 4)}
    print(f"\nSummary: {metrics}")

    # Persist metrics
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return y_true, y_pred


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    args = parse_args()





    print("\n" + "=" * 55)
    print("   Brain Tumor Detection — Training Pipeline")
    print("=" * 55)
    print(f"  Data dir    : {args.data_dir}")
    print(f"  Image size  : {IMG_SIZE}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Fine-tuning : {args.fine_tune}")
    print("=" * 55 + "\n")

    # ── Data ──────────────────────────────────
    train_gen, val_gen = build_generators(
        args.data_dir, args.val_split, args.batch_size
    )
    class_names = list(train_gen.class_indices.keys())
    class_weights = compute_class_weights(train_gen)
    print(f"[Data] Classes detected: {train_gen.class_indices}")
    print(f"[Data] Val class indices:   {val_gen.class_indices}")
    print(f"[Data] Class weights:       {class_weights}")
    print(f"[Data] Train samples: {train_gen.samples}, "
          f"Val samples: {val_gen.samples}")
    assert train_gen.class_indices == val_gen.class_indices, (
        "Train/val class index mismatch — check generator setup."
    )

    # Persist class label mapping
    os.makedirs("models", exist_ok=True)
    with open(LABELS_PATH, "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)

    # ── Phase 1: Train head only ───────────────
    model, base = build_model(NUM_CLASSES, freeze_base=True)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    phase1_epochs = max(1, args.epochs // 2)
    print(f"\n[Phase 1] Training classification head ({phase1_epochs} epochs)...")
    hist1 = model.fit(
        train_gen,
        epochs=phase1_epochs,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=build_callbacks(phase=1),
        verbose=1,
    )

    # ── Phase 2 (optional): fine-tune top layers ─
    if args.fine_tune:
        unfreeze_top_layers(model, base, num_layers_to_unfreeze=30)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-5),  # much smaller LR
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        phase2_epochs = args.epochs - phase1_epochs
        print(f"\n[Phase 2] Fine-tuning top base layers ({phase2_epochs} epochs)...")
        hist2 = model.fit(
            train_gen,
            epochs=phase2_epochs,
            validation_data=val_gen,
            class_weight=class_weights,
            callbacks=build_callbacks(phase=2),
            verbose=1,
        )
        # Merge histories
        for k in hist1.history:
            hist1.history[k].extend(hist2.history.get(k, []))

    # ── Evaluate (ModelCheckpoint already saved best weights) ──
    print(f"\n[OK] Best model weights -> {MODEL_PATH}")

    save_training_curves(hist1)
    y_true, y_pred = evaluate_model(model, val_gen, class_names)
    save_confusion_matrix(y_true, y_pred, class_names)

    print("\n[OK] Training complete. All artifacts saved to results/ and models/")
    print(train_gen.class_indices)

if __name__ == "__main__":
    main()
