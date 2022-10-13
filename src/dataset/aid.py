"""
AID: A scene classification dataset
"""
import glob
import os
from pathlib import Path
from typing import Dict, Tuple

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
        self.train_iterator = self._build_iterator('training')
        self.validation_iterator = self._build_iterator('validation')

    def _build_iterator(self, dataset_subset: str):
        dataset = tf.keras.utils.image_dataset_from_directory(
            self.dataset_path, batch_size=self.batch_size, seed=42,
            image_size=self.image_size, validation_split=self.validation_split, subset=dataset_subset)
        return dataset

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
    aid = AID('datasets/AID/', batch_size=8, image_size=(200, 200))
