# Model Card: Brain Tumor MRI Classifier

## Model Overview

| Field | Value |
|---|---|
| **Model name** | BrainTumorDetector |
| **Architecture** | EfficientNetB0 (ImageNet) + custom dense head |
| **Task** | Binary MRI slice classification (tumor / no tumor) |
| **Framework** | TensorFlow / Keras 3 |
| **Input** | RGB image, 224x224, float32 pixel values in **[0, 255]** |
| **Output** | Softmax over 2 classes: `no` (index 0), `yes` (index 1) |

## Intended Use

**Primary use:** Academic research and portfolio demonstration of medical image classification with explainability (Grad-CAM).

**Out of scope:**
- Clinical diagnosis or treatment decisions
- Real-time surgical guidance
- DICOM PACS integration (JPEG/PNG only in current version)
- Pediatric, post-contrast, or multi-sequence MRI protocols without retraining

## Training Data

| Source | [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |
|---|---|
| **Classes** | `yes` (tumor present), `no` (no tumor) |
| **Approx. size** | 253 images (155 yes / 98 no) |
| **Format** | JPEG/PNG axial MRI slices |
| **Split** | 80/20 stratified hold-out (seed=42) or 5-fold CV via `scripts/cross_validate.py` |

Class imbalance is handled with `sklearn` balanced class weights during training.

## Preprocessing (Critical)

Keras 3 EfficientNetB0 includes internal `Rescaling(1/255)`. **Do not divide pixels by 255 before inference.** Inputs must remain in `[0, 255]` as float32.

## Evaluation Metrics

Report on the validation hold-out after full training (example run):

| Metric | Description |
|---|---|
| **Accuracy** | Overall correct classifications |
| **Sensitivity (Recall)** | True positive rate for tumor class |
| **Specificity** | True negative rate for no-tumor class |
| **AUC-ROC** | Discrimination across thresholds |
| **F1 (weighted)** | Harmonic mean with class weighting |

Metrics are saved to `results/metrics.json`. ROC curves: `results/roc_curve.png`.

Cross-validation summary: `results/cv_metrics.json`.

## Explainability

Grad-CAM (Selvaraju et al., ICCV 2017) highlights spatial regions influencing the prediction. Warm regions indicate higher activation. Radiologists should treat heatmaps as supplementary — not ground truth.

## Limitations and Risks

1. **Small dataset** — High variance; external validation is required before any real-world use.
2. **Single split metrics** — Can be optimistic; prefer k-fold CV results in `cv_metrics.json`.
3. **Class imbalance** — Model may bias toward majority class if preprocessing or weights are misconfigured.
4. **Domain shift** — Scanner, sequence, and institution differences will degrade performance.
5. **No calibration analysis** — Reported confidence is raw softmax probability, not calibrated clinical probability.
6. **JPEG artifacts** — Compression may affect saliency maps and predictions.

See [docs/LIMITATIONS.md](docs/LIMITATIONS.md) for extended discussion.

## Ethical Considerations

- De-identified public dataset; no patient identifiers stored in this repository.
- System must not replace qualified neuroradiologist review.
- False negatives (missed tumors) carry serious health risks in clinical settings.

## Reproducibility

- Config: `configs/default.yaml`
- Seed: 42 (TensorFlow + NumPy)
- Dependencies: `requirements.txt` (pinned versions)
- Train: `python train.py --config configs/default.yaml --fine_tune`

## Citation

If you use this dataset, cite the original Kaggle dataset and:

```
Sohail, K. (2026). Brain Tumor MRI Classification with EfficientNetB0 and Grad-CAM.
GitHub repository: brain-tumor-detection.
```

## Changelog

| Version | Notes |
|---|---|
| 1.0.0 | Initial release; fixed double-rescaling bug; added CV, ROC, model card |
