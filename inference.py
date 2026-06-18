"""
inference.py — Brain Tumor Detector: Prediction Module
=======================================================
Loads the trained model, preprocesses an input MRI image, runs inference,
and returns the prediction, confidence score, and Grad-CAM visualisation.

Usage (from project root):
    python -c "from app.inference import predict; predict('path/to/mri.jpg')"

Or via predict.py:
    python predict.py --image path/to/mri.jpg
"""

import os
import json
import numpy as np
import cv2
import tensorflow as tf

# Lazy import: gradcam_utils may not be on path when running predict.py directly
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gradcam_utils import generate_gradcam, plot_gradcam_result

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
IMG_SIZE    = (224, 224)
MODEL_PATH  = "models/brain_tumor_model.h5"
LABELS_PATH = "models/class_labels.json"

_model_cache       = None   # singleton model cache
_class_names_cache = None


# ──────────────────────────────────────────────
# Model loading (singleton)
# ──────────────────────────────────────────────
def load_model(model_path=MODEL_PATH):
    """
    Load the trained Keras model once and cache it in memory.
    Subsequent calls return the cached model.
    """
    global _model_cache
    if _model_cache is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at '{model_path}'. "
                "Run train.py first to generate the model weights."
            )
        print(f"[Model] Loading from {model_path} ...")
        _model_cache = tf.keras.models.load_model(model_path)
        print("[Model] Loaded successfully.")
    return _model_cache


def load_class_names(labels_path=LABELS_PATH):
    """
    Load class index → class name mapping from the JSON saved during training.
    Returns a list ordered by index: ['no', 'yes'] or ['yes', 'no'].
    """
    global _class_names_cache
    if _class_names_cache is None:
        if not os.path.exists(labels_path):
            # Fallback defaults (alphabetical, matches Keras flow_from_directory)
            _class_names_cache = ["no", "yes"]
        else:
            with open(labels_path) as f:
                label_map = json.load(f)  # {"no": 0, "yes": 1}
            # Invert: {0: "no", 1: "yes"}
            inv = {v: k for k, v in label_map.items()}
            _class_names_cache = [inv[i] for i in range(len(inv))]
    return _class_names_cache


# ──────────────────────────────────────────────
# Image preprocessing
# ──────────────────────────────────────────────
def load_and_preprocess(image_path):
    """
    Load an MRI image from disk and prepare it for model input.

    Returns
    -------
    img_array      : np.ndarray, shape (1, 224, 224, 3), float32 in [0, 255]
    original_uint8 : np.ndarray, shape (224, 224, 3), uint8 — for overlays
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Read with OpenCV (BGR), convert to RGB
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not decode image: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Resize to model input size
    resized = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_LANCZOS4)
    original_uint8 = resized.copy()

    # Keep pixel values in [0, 255] — EfficientNetB0 rescales internally.
    # Do NOT divide by 255 (that causes double-rescaling and ~50/50 outputs).
    img_array = resized.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

    return img_array, original_uint8


# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────
def predict(image_path, save_gradcam=True, gradcam_save_path=None,
            model_path=MODEL_PATH, labels_path=LABELS_PATH,
            show_plot=False):
    """
    Full prediction pipeline for a single MRI image.

    Parameters
    ----------
    image_path       : str — path to the MRI image file
    save_gradcam     : bool — whether to save the Grad-CAM visualisation
    gradcam_save_path: str or None — custom path for the output figure;
                       defaults to results/gradcam_<filename>.png
    model_path       : str
    labels_path      : str
    show_plot        : bool — display plot interactively (set False on servers)

    Returns
    -------
    result : dict with keys:
        'prediction'   : str — 'Tumor' or 'No Tumor'
        'confidence'   : float — probability of the predicted class
        'class_index'  : int — 0 or 1
        'all_probs'    : np.ndarray — softmax probabilities for all classes
        'gradcam_path' : str or None — path to the saved Grad-CAM figure
        'overlay'      : np.ndarray — the overlay image (uint8)
        'heatmap_rgb'  : np.ndarray — colourised heatmap (uint8)
        'original_img' : np.ndarray — preprocessed original (uint8)
    """
    model       = load_model(model_path)
    class_names = load_class_names(labels_path)

    # Preprocess
    img_array, original_uint8 = load_and_preprocess(image_path)

    # Inference
    probs      = model.predict(img_array, verbose=0)[0]  # shape: (num_classes,)
    pred_index = int(np.argmax(probs))
    confidence = float(probs[pred_index])

    # Map to human-readable label (index order matches class_labels.json)
    raw_label = class_names[pred_index]      # e.g. 'yes' or 'no'
    label     = "Tumor" if raw_label.lower() in ("yes", "1", "tumor") else "No Tumor"

    # Grad-CAM
    gradcam = generate_gradcam(
        model, img_array, original_uint8, pred_index=pred_index
    )

    # Determine save path
    if save_gradcam:
        if gradcam_save_path is None:
            base = os.path.splitext(os.path.basename(image_path))[0]
            gradcam_save_path = os.path.join("results", f"gradcam_{base}.png")
        os.makedirs(os.path.dirname(gradcam_save_path) or ".", exist_ok=True)
        fig = plot_gradcam_result(
            original_uint8, gradcam["overlay"], gradcam["heatmap_rgb"],
            label, confidence, save_path=gradcam_save_path
        )
        if show_plot:
            import matplotlib.pyplot as plt
            plt.show()
        import matplotlib.pyplot as plt
        plt.close(fig)
    else:
        gradcam_save_path = None

    result = {
        "prediction":   label,
        "confidence":   confidence,
        "class_index":  pred_index,
        "all_probs":    probs,
        "gradcam_path": gradcam_save_path,
        "overlay":      gradcam["overlay"],
        "heatmap_rgb":  gradcam["heatmap_rgb"],
        "original_img": original_uint8
    }

    # ── Console summary ───────────────────────
    border = "─" * 44
    print(f"\n{border}")
    print(f"  BRAIN TUMOR DETECTION — RESULT")
    print(f"{border}")
    print(f"  Image      : {os.path.basename(image_path)}")
    print(f"  Prediction : {label}")
    print(f"  Confidence : {confidence:.2%}")
    for i, (name, prob) in enumerate(zip(class_names, probs)):
        bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
        print(f"  {name:>8}  [{bar}]  {prob:.2%}")
    if gradcam_save_path:
        print(f"  Grad-CAM   : {gradcam_save_path}")
    print(f"{border}\n")

    return result


# ──────────────────────────────────────────────
# Batch prediction
# ──────────────────────────────────────────────
def predict_batch(image_paths, **kwargs):
    """
    Run predict() on a list of images.
    Returns a list of result dicts.
    """
    results = []
    for path in image_paths:
        try:
            r = predict(path, **kwargs)
            results.append(r)
        except Exception as e:
            print(f"[!] Failed on {path}: {e}")
            results.append({"error": str(e), "image_path": path})
    return results
