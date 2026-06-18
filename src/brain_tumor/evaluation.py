"""Metrics, plots, validation evaluation, and cross-validation."""

from __future__ import annotations

import json
import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

from brain_tumor.data import collect_labeled_paths


def predict_generator(model, generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator.reset()
    y_true_batches, y_prob_batches = [], []
    for batch_idx in range(len(generator)):
        batch_x, batch_y = generator[batch_idx]
        y_prob_batches.append(model.predict(batch_x, verbose=0))
        y_true_batches.append(np.argmax(batch_y, axis=1))
    y_prob = np.vstack(y_prob_batches)
    y_true = np.concatenate(y_true_batches)
    y_pred = np.argmax(y_prob, axis=1)
    return y_true, y_pred, y_prob


def medical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    positive_index: int = 1,
) -> dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    acc = float(np.mean(y_true == y_pred))
    prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    scores = y_prob[:, positive_index]
    binary = (y_true == positive_index).astype(int)
    if len(np.unique(binary)) > 1:
        fpr, tpr, _ = roc_curve(binary, scores)
        roc_auc = float(auc(fpr, tpr))
    else:
        fpr, tpr, roc_auc = np.array([0, 1]), np.array([0, 1]), 0.0

    return {
        "accuracy": round(acc, 4),
        "precision_weighted": round(prec, 4),
        "recall_weighted": round(rec, 4),
        "f1_weighted": round(f1, 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "auc_roc": round(roc_auc, 4),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "_roc_fpr": fpr.tolist(),
        "_roc_tpr": tpr.tolist(),
    }


def save_training_curves(history, filename: str) -> None:
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Diagnostics — Brain Tumor Detector", fontsize=14, fontweight="bold")

    axes[0].plot(history.history["accuracy"], label="Train Acc", color="#2196F3", lw=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Acc", color="#FF5722", lw=2, ls="--")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"], label="Train Loss", color="#4CAF50", lw=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss", color="#9C27B0", lw=2, ls="--")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Categorical Cross-Entropy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Training curves saved -> {filename}")


def save_confusion_matrix_plot(
    y_true, y_pred, class_names, filename: str
) -> None:
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix — Brain Tumor Detection", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Confusion matrix saved -> {filename}")


def save_roc_curve(metrics: dict[str, Any], filename: str, title: str = "ROC Curve") -> None:
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    fpr = metrics.pop("_roc_fpr", [0, 1])
    tpr = metrics.pop("_roc_tpr", [0, 1])
    auc_val = metrics.get("auc_roc", 0.0)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"AUC = {auc_val:.3f}")
    ax.plot([0, 1], [0, 1], color="#64748b", ls="--", lw=1)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] ROC curve saved -> {filename}")


def evaluate_model(
    model,
    val_gen,
    class_names: list[str],
    results_dir: str,
    positive_index: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    print("\n[Eval] Running inference on validation set...")
    y_true, y_pred, y_prob = predict_generator(model, val_gen)

    print("\n" + "=" * 55)
    print("         CLASSIFICATION REPORT")
    print("=" * 55)
    print(classification_report(y_true, y_pred, target_names=class_names))

    metrics = medical_metrics(y_true, y_pred, y_prob, positive_index=positive_index)
    public_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}
    print(f"\nSummary: {public_metrics}")

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(public_metrics, f, indent=2)

    save_roc_curve(dict(metrics), os.path.join(results_dir, "roc_curve.png"))
    return y_true, y_pred, public_metrics


def run_cross_validation(
    cfg: dict[str, Any],
    train_fn,
    data_dir: str,
    n_splits: int = 5,
) -> dict[str, Any]:
    """
    Stratified k-fold CV. train_fn(fold_idx, train_paths, train_labels, val_paths, val_labels)
    must return (y_true, y_pred, y_prob) on the validation fold.
    """
    paths, labels, class_indices = collect_labeled_paths(data_dir)
    labels = np.array(labels)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg["seed"])

    fold_metrics = []
    all_y_true, all_y_pred, all_y_prob = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(paths, labels)):
        print(f"\n[CV] Fold {fold + 1}/{n_splits}")
        y_true, y_pred, y_prob = train_fn(
            fold,
            [paths[i] for i in train_idx],
            labels[train_idx].tolist(),
            [paths[i] for i in val_idx],
            labels[val_idx].tolist(),
            class_indices,
        )
        m = medical_metrics(
            y_true,
            y_pred,
            y_prob,
            positive_index=cfg["evaluation"]["positive_class_index"],
        )
        public = {k: v for k, v in m.items() if not k.startswith("_")}
        public["fold"] = fold + 1
        fold_metrics.append(public)
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)
        all_y_prob.append(y_prob)

    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)
    all_y_prob = np.vstack(all_y_prob)

    aggregate = medical_metrics(
        all_y_true,
        all_y_pred,
        all_y_prob,
        positive_index=cfg["evaluation"]["positive_class_index"],
    )
    summary_keys = [
        "accuracy", "sensitivity", "specificity", "auc_roc", "f1_weighted"
    ]
    summary = {}
    for key in summary_keys:
        values = [f[key] for f in fold_metrics]
        summary[f"{key}_mean"] = round(float(np.mean(values)), 4)
        summary[f"{key}_std"] = round(float(np.std(values)), 4)

    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    out = {
        "n_splits": n_splits,
        "class_indices": class_indices,
        "folds": fold_metrics,
        "summary": summary,
        "pooled": {k: v for k, v in aggregate.items() if not k.startswith("_")},
    }
    with open(os.path.join(results_dir, "cv_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    inv = {v: k for k, v in class_indices.items()}
    class_names = [inv[i] for i in range(len(inv))]
    save_confusion_matrix_plot(
        all_y_true,
        all_y_pred,
        class_names,
        os.path.join(results_dir, "cv_confusion_matrix.png"),
    )
    save_roc_curve(
        dict(aggregate),
        os.path.join(results_dir, "cv_roc_curve.png"),
        title="Pooled Cross-Validation ROC",
    )
    print(f"\n[OK] Cross-validation summary: {summary}")
    return out
