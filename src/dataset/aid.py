"""
AID: A scene classification dataset
"""
import glob
import os
from pathlib import Path
from typing import Dict, Tuple

import tensorflow as tf
import tensorflow.python.keras.layers as layers
from tensorflow.keras import Sequential

from src.dataset.balancing import compute_class_weight


class AID:

    def __init__(self, dataset_path: str, batch_size: int, image_size: Tuple[int, int],
                 validation_split: float = 0.1):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.validation_split = validation_split

        self.class_weights = self._compute_class_weights()
        self.train_iterator, self.train_steps = self._build_iterator('training', augment=True)
        self.validation_iterator, self.validation_steps = self._build_iterator('validation', augment=False)

    def _build_iterator(self, dataset_subset: str, augment: bool):
        augmentation_pipe = self._get_augmentation_pipeline()

        dataset = tf.keras.utils.image_dataset_from_directory(
            self.dataset_path, batch_size=self.batch_size, seed=42,
            image_size=self.image_size, validation_split=self.validation_split, subset=dataset_subset)
        dataset_batches = dataset.cardinality().numpy()
        dataset = dataset.repeat()
        if augment:
            dataset = dataset.map(lambda x, y: (augmentation_pipe(x), y),
                                  num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset, dataset_batches

    def _get_augmentation_pipeline(self):
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

    def _compute_class_weights(self) -> Dict:
        image_counts = {}
        subdirs = glob.glob(os.path.join(self.dataset_path, "*"))
        for subdir in subdirs:
            label = Path(subdir).stem
            files = glob.glob(os.path.join(subdir, "*"))
            image_counts[label] = len(files)
        class_weight = compute_class_weight(image_counts)
        return class_weight


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use('TkAgg')

    aid = AID('datasets/AID/', batch_size=32, image_size=(200, 200))
    for im, label in aid.train_iterator:
        plt.imshow(im[0].numpy().astype(int))
        plt.show()
        pass
