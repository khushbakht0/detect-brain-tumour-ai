# Limitations and Future Work

## Current Limitations

### Data
- Only 253 labeled slices from a single public dataset.
- Moderate class imbalance (155 tumor vs 98 no-tumor).
- No independent external test set from a different hospital or scanner.

### Model
- Transfer learning from ImageNet; MRI domain gap remains.
- Binary slice-level classification — not volumetric (3D) analysis.
- No segmentation of tumor boundaries.

### Engineering
- Model weights are not committed to git; users must train locally or supply weights for deployment.
- DICOM format is not supported; users must export slices to PNG/JPEG.
- No model versioning registry (MLflow/W&B) in the default pipeline.

### Regulatory
- This is a **research prototype**, not a medical device.
- Not validated under FDA, CE, or local clinical software regulations.

## Why Grad-CAM Matters

Grad-CAM provides post-hoc explanations showing where the network attended. In medical AI portfolios, this demonstrates awareness that:

- High accuracy alone is insufficient for clinical trust.
- Models can focus on spurious correlates (scanner artifacts, borders, text overlays).
- Human review of saliency maps is a reasonable first step toward safe deployment.

## Future Work

1. **Stratified k-fold ensembling** — Already supported via `scripts/cross_validate.py`; extend to model averaging.
2. **DICOM pipeline** — `pydicom` ingestion with windowing and orientation normalization.
3. **3D CNN or slice aggregation** — Combine multiple slices per study.
4. **Probability calibration** — Temperature scaling or Platt scaling on a held-out set.
5. **External validation** — Second dataset with different acquisition parameters.
6. **Quantitative XAI metrics** — Deletion/insertation tests for Grad-CAM faithfulness.

## Class Imbalance Handling

Training uses `sklearn.utils.class_weight.compute_class_weight(balanced)` so the loss function penalizes errors on the minority class (`no`) more heavily. Document any change to this strategy in `MODEL_CARD.md` when experimenting.

## Preprocessing Bug (Historical)

Earlier versions divided inputs by 255 while EfficientNetB0 also rescales internally, producing near-black images and collapsed ~50/50 predictions. Current code keeps pixels in `[0, 255]`. Tests in `tests/test_preprocessing.py` guard against regression.
