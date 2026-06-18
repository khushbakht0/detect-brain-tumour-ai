"""Inference and preprocessing for brain tumor classification."""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
import tensorflow as tf

from brain_tumor.config import load_config, resolve_path
from brain_tumor.gradcam import generate_gradcam, plot_gradcam_result

_model_cache = None
_class_names_cache = None
_config_cache = None


def get_default_paths(cfg=None):
    cfg = cfg or load_config()
    return resolve_path(cfg, "model_path"), resolve_path(cfg, "labels_path")


def load_model(model_path: str | None = None, cfg=None):
    global _model_cache
    if _model_cache is None:
        if model_path is None:
            model_path, _ = get_default_paths(cfg)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at '{model_path}'. Run train.py first."
            )
        print(f"[Model] Loading from {model_path} ...")
        _model_cache = tf.keras.models.load_model(model_path)
        print("[Model] Loaded successfully.")
    return _model_cache


def load_class_names(labels_path: str | None = None, cfg=None) -> list[str]:
    global _class_names_cache
    if _class_names_cache is None:
        if labels_path is None:
            _, labels_path = get_default_paths(cfg)
        if not os.path.exists(labels_path):
            _class_names_cache = ["no", "yes"]
        else:
            with open(labels_path, encoding="utf-8") as f:
                label_map = json.load(f)
            inv = {v: k for k, v in label_map.items()}
            _class_names_cache = [inv[i] for i in range(len(inv))]
    return _class_names_cache


def preprocess_rgb_array(rgb: np.ndarray, img_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Resize and scale to float32 [0, 255] with batch dimension."""
    resized = cv2.resize(rgb, img_size, interpolation=cv2.INTER_LANCZOS4)
    original_uint8 = resized.copy()
    img_array = resized.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, original_uint8


def load_and_preprocess(image_path: str, img_size: tuple[int, int] = (224, 224)):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not decode image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return preprocess_rgb_array(rgb, img_size)


def raw_label_to_display(raw_label: str) -> str:
    return "Tumor" if raw_label.lower() in ("yes", "1", "tumor") else "No Tumor"


def predict(
    image_path: str,
    save_gradcam: bool = True,
    gradcam_save_path: str | None = None,
    model_path: str | None = None,
    labels_path: str | None = None,
    show_plot: bool = False,
    cfg=None,
):
    cfg = cfg or load_config()
    img_size = tuple(cfg["img_size"])
    model = load_model(model_path, cfg)
    class_names = load_class_names(labels_path, cfg)

    img_array, original_uint8 = load_and_preprocess(image_path, img_size)
    probs = model.predict(img_array, verbose=0)[0]
    pred_index = int(np.argmax(probs))
    confidence = float(probs[pred_index])
    label = raw_label_to_display(class_names[pred_index])

    gradcam = generate_gradcam(model, img_array, original_uint8, pred_index=pred_index)

    if save_gradcam:
        if gradcam_save_path is None:
            base = os.path.splitext(os.path.basename(image_path))[0]
            gradcam_save_path = os.path.join(
                resolve_path(cfg, "results_dir"), f"gradcam_{base}.png"
            )
        fig = plot_gradcam_result(
            original_uint8,
            gradcam["overlay"],
            gradcam["heatmap_rgb"],
            label,
            confidence,
            save_path=gradcam_save_path,
        )
        if show_plot:
            plt_show = __import__("matplotlib.pyplot", fromlist=["show"])
            plt_show.show()
        import matplotlib.pyplot as plt
        plt.close(fig)
    else:
        gradcam_save_path = None

    result = {
        "prediction": label,
        "confidence": confidence,
        "class_index": pred_index,
        "all_probs": probs,
        "gradcam_path": gradcam_save_path,
        "overlay": gradcam["overlay"],
        "heatmap_rgb": gradcam["heatmap_rgb"],
        "original_img": original_uint8,
    }

    border = "-" * 44
    print(f"\n{border}")
    print("  BRAIN TUMOR DETECTION — RESULT")
    print(border)
    print(f"  Image      : {os.path.basename(image_path)}")
    print(f"  Prediction : {label}")
    print(f"  Confidence : {confidence:.2%}")
    for name, prob in zip(class_names, probs):
        bar_len = int(prob * 20)
        bar = "#" * bar_len + "-" * (20 - bar_len)
        print(f"  {name:>8}  [{bar}]  {prob:.2%}")
    if gradcam_save_path:
        print(f"  Grad-CAM   : {gradcam_save_path}")
    print(f"{border}\n")
    return result


def predict_batch(image_paths, **kwargs):
    results = []
    for path in image_paths:
        try:
            results.append(predict(path, **kwargs))
        except Exception as exc:
            print(f"[!] Failed on {path}: {exc}")
            results.append({"error": str(exc), "image_path": path})
    return results
