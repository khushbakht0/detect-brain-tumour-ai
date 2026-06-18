"""Model architecture and fine-tuning helpers."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0


def build_model(
    img_size: tuple[int, int],
    num_classes: int = 2,
    freeze_base: bool = True,
) -> tuple[models.Model, models.Model]:
    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*img_size, 3),
    )
    base.trainable = not freeze_base

    inputs = tf.keras.Input(shape=(*img_size, 3))
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


def unfreeze_top_layers(model, base_model, num_layers_to_unfreeze: int = 30) -> None:
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
    trainable = sum(1 for layer in model.layers if layer.trainable)
    print(f"[Fine-tune] Unfroze top {num_layers_to_unfreeze} base layers.")
    print(f"[Fine-tune] Trainable layers: {trainable}")
