from tensorflow.keras import layers, models
import numpy as np

def build_model(input_shape, num_classes):

    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation="relu", input_shape=input_shape, padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model