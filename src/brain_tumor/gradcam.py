"""Grad-CAM explainability utilities."""

from __future__ import annotations

import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def _build_gradcam_model(model):
    conv_out = model.get_layer("efficientnetb0")(model.input)
    x = conv_out
    for layer in model.layers[2:]:
        x = layer(x)
    return tf.keras.Model(model.input, [conv_out, x], name="gradcam_model")


def get_gradcam_heatmap(model, img_array, pred_index=None):
    grad_model = _build_gradcam_model(model)

    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(inputs, training=False)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap_on_image(original_img, heatmap, alpha=0.45, colormap="jet"):
    h, w = original_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    cmap = cm.get_cmap(colormap)
    heatmap_rgba = (cmap(heatmap_resized) * 255).astype(np.uint8)
    heatmap_rgb = heatmap_rgba[:, :, :3]
    superimposed = cv2.addWeighted(heatmap_rgb, alpha, original_img, 1 - alpha, 0)
    return superimposed, heatmap_rgb


def generate_gradcam(model, img_array, original_img_uint8, pred_index=None, alpha=0.45):
    heatmap = get_gradcam_heatmap(model, img_array, pred_index)
    overlay, heatmap_rgb = overlay_heatmap_on_image(
        original_img_uint8, heatmap, alpha=alpha
    )
    return {"heatmap": heatmap, "overlay": overlay, "heatmap_rgb": heatmap_rgb}


def plot_gradcam_result(original_img, overlay, heatmap_rgb, label, confidence, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0d0d0d")
    titles = ["Original MRI", "Grad-CAM Heatmap", "Overlay (Explainability)"]
    for ax, img, title in zip(axes, [original_img, heatmap_rgb, overlay], titles):
        ax.imshow(img)
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")
        ax.set_facecolor("#0d0d0d")

    color = "#FF4B4B" if label == "Tumor" else "#4CAF50"
    fig.text(
        0.5, 0.01,
        f"Prediction: {label}   |   Confidence: {confidence:.1%}",
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=color,
        bbox=dict(facecolor="#1a1a1a", edgecolor=color, boxstyle="round,pad=0.4"),
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[OK] Grad-CAM figure saved -> {save_path}")
    return fig
