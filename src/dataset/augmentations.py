from tensorflow.python.keras import layers, Sequential


def get_augmentation_pipeline():
    augmentation_pipeline = Sequential([
        layers.RandomContrast(0.2),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomZoom(
            height_factor=(-0.2, 0.2),
            width_factor=(-0.2, 0.2)),
        layers.RandomRotation(1.0),  # rotate randomly in the 360 degree range
        layers.RandomTranslation(
            height_factor=(-0.2, 0.2),
            width_factor=(-0.2, 0.2)),
    ])
    return augmentation_pipeline