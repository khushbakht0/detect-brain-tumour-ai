"""
gradcam_utils.py — Gradient-weighted Class Activation Mapping (Grad-CAM)
=========================================================================
Produces pixel-level saliency maps that highlight which regions of an MRI
influenced the model's prediction. This provides clinical explainability —
a radiologist can verify that the model focused on anatomically plausible
regions rather than image artifacts.

Reference:
    Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep
    Networks via Gradient-based Localization." ICCV 2017.
    https://arxiv.org/abs/1610.02391
"""

import numpy as np
import cv2
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ──────────────────────────────────────────────
# Core Grad-CAM computation
# ──────────────────────────────────────────────
def _build_gradcam_model(model):
    """
    Build a sub-model (conv features, predictions) from the main model graph.
    Keras 3 nested Functional models break tf.keras.Model(..., layer.output);
    rebuilding from model.input keeps the graph connected for GradientTape.
    """
    conv_out = model.get_layer("efficientnetb0")(model.input)
    x = conv_out
    for layer in model.layers[2:]:
        x = layer(x)
    return tf.keras.Model(model.input, [conv_out, x], name="gradcam_model")


def get_gradcam_heatmap(model, img_array, conv_layer_ref=None, pred_index=None):
    """
    Compute Grad-CAM heatmap for a single image.

    Parameters
    ----------
    model             : tf.keras.Model — the trained classifier
    img_array         : np.ndarray, shape (1, H, W, 3), float32 in [0, 255]
    conv_layer_ref    : unused — kept for API compatibility
    pred_index        : int or None — class index to explain.
                        If None, uses the top predicted class.

    Returns
    -------
    heatmap : np.ndarray, shape (H', W'), float32 in [0, 1]
    """
    grad_model = _build_gradcam_model(model)

    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(inputs, training=False)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    # Global average pooling of gradients → importance weight per filter
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the feature maps by the pooled gradients
    conv_outputs = conv_outputs[0]                          # (H', W', C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (H', W', 1)
    heatmap = tf.squeeze(heatmap)                           # (H', W')

    # ReLU: keep only positive activations (regions supporting the class)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def find_last_conv_layer(model):
    """
    Auto-detect the name of the last convolutional layer in the model.
    Works for EfficientNetB0 and MobileNetV2 feature extractors.
    """
    for layer in reversed(model.layers):
        # Check the layer itself
        if isinstance(layer, (tf.keras.layers.Conv2D,
                               tf.keras.layers.DepthwiseConv2D)):
            return layer.name
        # Recurse into sub-models (e.g., the EfficientNetB0 base)
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, (tf.keras.layers.Conv2D,
                                     tf.keras.layers.DepthwiseConv2D)):
                    # Return name scoped to parent model
                    return layer.name
    raise ValueError("Could not find a Conv2D layer in the model.")


def find_efficientnet_last_conv(model):
    """
    EfficientNetB0-specific helper: returns (base_model_name, last_conv_layer_name)
    for use with get_gradcam_heatmap on nested Keras 3 models.
    """
    base_model = None
    for layer in model.layers:
        if "efficientnet" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        return find_last_conv_layer(model)

    for layer in reversed(base_model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D,
                               tf.keras.layers.DepthwiseConv2D)):
            return base_model.name, layer.name

    raise ValueError("Cannot locate conv layer in EfficientNetB0 base.")


# ──────────────────────────────────────────────
# Overlay helpers
# ──────────────────────────────────────────────
def overlay_heatmap_on_image(original_img, heatmap, alpha=0.45, colormap="jet"):
    """
    Superimpose the Grad-CAM heatmap on the original MRI image.

    Parameters
    ----------
    original_img : np.ndarray, uint8 (H, W, 3)
    heatmap      : np.ndarray, float32 (H', W') in [0, 1]
    alpha        : float — blending factor (higher = more heatmap visible)
    colormap     : str — matplotlib colormap name

    Returns
    -------
    superimposed : np.ndarray, uint8 (H, W, 3)
    heatmap_rgb  : np.ndarray, uint8 (H, W, 3)  — colourised heatmap only
    """
    h, w = original_img.shape[:2]

    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Apply colourmap and convert to uint8 RGB
    cmap       = cm.get_cmap(colormap)
    heatmap_rgba = (cmap(heatmap_resized) * 255).astype(np.uint8)
    heatmap_rgb  = heatmap_rgba[:, :, :3]  # drop alpha channel

    # Blend: alpha * heatmap + (1 - alpha) * original
    superimposed = cv2.addWeighted(
        heatmap_rgb, alpha,
        original_img, 1 - alpha,
        0
    )
    return superimposed, heatmap_rgb


def generate_gradcam(model, img_array, original_img_uint8,
                     pred_index=None, alpha=0.45):
    """
    End-to-end Grad-CAM: heatmap → overlay.

    Parameters
    ----------
    model              : trained tf.keras.Model
    img_array          : (1, 224, 224, 3) float32 in [0, 255]
    original_img_uint8 : (224, 224, 3) uint8 for overlay
    pred_index         : int or None
    alpha              : blending factor

    Returns
    -------
    dict with keys:
        'heatmap'     : raw float heatmap
        'overlay'     : RGB overlay (uint8)
        'heatmap_rgb' : colourised heatmap (uint8)
    """
    # Get spatial feature maps from EfficientNet (7x7x1280 before pooling).
    last_conv = "efficientnetb0"

    heatmap  = get_gradcam_heatmap(model, img_array, last_conv, pred_index)
    overlay, heatmap_rgb = overlay_heatmap_on_image(
        original_img_uint8, heatmap, alpha=alpha
    )
    return {
        "heatmap":     heatmap,
        "overlay":     overlay,
        "heatmap_rgb": heatmap_rgb
    }


# ──────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────
def plot_gradcam_result(original_img, overlay, heatmap_rgb,
                         label, confidence, save_path=None):
    """
    Three-panel figure: original | heatmap | overlay.
    Adds confidence bar and prediction label.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0d0d0d")

    titles = ["Original MRI", "Grad-CAM Heatmap", "Overlay (Explainability)"]
    images = [original_img, heatmap_rgb, overlay]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")
        ax.set_facecolor("#0d0d0d")

    # Prediction banner
    color = "#FF4B4B" if label == "Tumor" else "#4CAF50"
    fig.text(
        0.5, 0.01,
        f"Prediction: {label}   |   Confidence: {confidence:.1%}",
        ha="center", va="bottom",
        fontsize=13, fontweight="bold",
        color=color,
        bbox=dict(facecolor="#1a1a1a", edgecolor=color, boxstyle="round,pad=0.4")
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[OK] Grad-CAM figure saved -> {save_path}")

    return fig


def save_gradcam_examples(model, image_paths, class_names,
                           output_path="results/gradcam_examples.png"):
    """
    Generate a multi-row Grad-CAM grid for a list of images.
    Useful for the research report figures.

    Parameters
    ----------
    model        : trained tf.keras.Model
    image_paths  : list of str — paths to sample MRI images
    class_names  : list of str — ['no', 'yes'] or similar
    output_path  : str
    """
    from inference import load_and_preprocess

    n = len(image_paths)
    fig, axes = plt.subplots(n, 3, figsize=(13, 4 * n))
    fig.patch.set_facecolor("#111111")

    if n == 1:
        axes = [axes]  # Ensure iterable

    col_titles = ["Original MRI", "Grad-CAM", "Overlay"]
    for col, ct in enumerate(col_titles):
        axes[0][col].set_title(ct, color="white", fontsize=11,
                                fontweight="bold", pad=10)

    for i, path in enumerate(image_paths):
        img_array, original_uint8 = load_and_preprocess(path)
        preds = model.predict(img_array, verbose=0)[0]
        pred_idx   = int(np.argmax(preds))
        confidence = float(preds[pred_idx])
        label      = class_names[pred_idx].upper()

        gradcam = generate_gradcam(model, img_array, original_uint8,
                                    pred_index=pred_idx)

        panels = [original_uint8, gradcam["heatmap_rgb"], gradcam["overlay"]]
        for j, panel in enumerate(panels):
            axes[i][j].imshow(panel)
            axes[i][j].axis("off")
            axes[i][j].set_facecolor("#111111")

        # Row label
        color = "#FF4B4B" if label == "TUMOR" else "#4CAF50"
        axes[i][0].set_ylabel(
            f"{label}\n{confidence:.1%}", color=color,
            fontsize=10, fontweight="bold", rotation=0, labelpad=60
        )

    plt.tight_layout()
    import os; os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[OK] Grad-CAM examples grid saved -> {output_path}")
