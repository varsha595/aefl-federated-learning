"""
model.py — the AEFL CNN architecture (trained from scratch, no pretrained
backbone, so federated averaging genuinely has something to learn) plus the
FedProx proximal-loss variant.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def create_model(image_size=128, learning_rate=0.001):
    """Custom lightweight CNN: 4 conv blocks + a small MLP head."""
    inputs = layers.Input(shape=(image_size, image_size, 3))

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="AEFL_CNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_weights(model):
    return model.get_weights()


def set_weights(model, weights):
    model.set_weights(weights)


def model_size_bytes(model):
    """Total size of trainable+non-trainable weights, assuming float32."""
    return int(sum(w.size for w in model.get_weights()) * 4)


def clone_with_weights(weights, image_size=128, learning_rate=0.001):
    model = create_model(image_size, learning_rate)
    model.set_weights(weights)
    return model


def proximal_term(local_weights, global_weights, mu):
    """FedProx regularizer: (mu/2) * ||w - w_global||^2, summed over layers."""
    term = 0.0
    for w_local, w_global in zip(local_weights, global_weights):
        term += tf.reduce_sum(tf.square(w_local - tf.constant(w_global)))
    return (mu / 2.0) * term
