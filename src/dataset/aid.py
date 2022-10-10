"""
AID: A scene classification dataset
"""
import glob
import os
from typing import Tuple

import numpy as np
import tensorflow as tf

from src.dataset.balancing import compute_class_weight


class AID:

    def __init__(self, dataset_path: str, batch_size: int, image_size: Tuple[int, int],
                 validation_split: float = 0.1):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.validation_split = validation_split

        self.class_weights = self._compute_class_weights()
        self.train_iterator, self.validation_iterator = self._build_iterator()

    def _build_iterator(self):
        dataset = tf.keras.utils.image_dataset_from_directory(
            self.dataset_path, batch_size=self.batch_size,
            image_size=self.image_size, validation_split=self.validation_split)
        return dataset

    def _compute_class_weights(self) -> np.ndarray:
        image_counts = []
        subdirs = glob.glob(os.path.join(self.dataset_path, "*"))
        for subdir in subdirs:
            files = glob.glob(os.path.join(subdir, "*"))
            image_counts.append(len(files))
        return compute_class_weight(image_counts)


if __name__ == "__main__":
    aid = AID('datasets/AID/', batch_size=8, image_size=(200, 200))
