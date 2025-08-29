from tensorflow.keras import layers, models
import numpy as np

def build_model(input_shape, num_classes):

    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model


def build_model(num_classes):
    import keras.backend as K

    K.clear_session()

    model = models.Sequential()
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # , input_shape=(8, 11, 1) # padding='same',
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(64, (1, 1), activation='relu'))
    # model.add(layers.LocallyConnected2D(64, (1, 1), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (1, 1), activation='relu'))
    # model.add(layers.LocallyConnected2D(64, (1, 1), activation='relu'))
    model.add(layers.BatchNormalization())

    # model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    # model.add(layers.Dropout(0.5))

    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    return model


def build_model_1D(input_shape, num_classes):

    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(16, 3, activation="relu", padding="same"),
        layers.MaxPooling1D(2),
        layers.Conv1D(32, 3, activation="relu", padding="same"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model