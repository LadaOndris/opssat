import glob
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from scipy.interpolate import NearestNDInterpolator


def get_images_from_path(dataset_path, input_shape):
    """ Get images from path and normalize them applying channel-level normalization. """

    # loading all images in one large batch
    tf_eval_data = tf.keras.utils.image_dataset_from_directory(dataset_path, image_size=input_shape[:2], shuffle=False,
                                                               batch_size=100000)

    # extract images and targets
    for tf_eval_images, tf_eval_targets in tf_eval_data:
        break

    return tf.convert_to_tensor(tf_eval_images), tf_eval_targets


class TrainDataset:
    """
    A class providing a tf.data.Dataset instance, which
    prepares data for training.
    """

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

        self.annotations = self._load_annotations()
        self._fill_unannotated_minitiles()
        self._check_annotations()


    def _load_annotations(self):
        """
        Reads existing .annot.npy files and keeps the arrays in memory.
        """
        # Find all annotation files
        pattern = os.path.join(self.dataset_path, '*.annot.npy')
        annot_files = glob.glob(pattern)
        # Read annotations
        annotations = {}
        for annot_file in annot_files:
            # Load annotations
            annotation_content = np.load(annot_file)
            # Get image file path
            path = Path(annot_file)
            file_name_stem = path.name.split('.')[0]
            image_path = os.path.join(path.parent, f'{file_name_stem}.png')
            # Assign the annotation to the image file
            annotations[image_path] = annotation_content
        return annotations

    def _fill_unannotated_minitiles(self):
        """
        Manual annotation is time-consuming. It is handy to annotate only borders
        of larger regions. This function then fills the unannotated minitiles by
        looking at the nearest annotated minitile and asigning the same type.
        """
        for image_path, annotation in self.annotations.items():
            mask = np.where(~(annotation == -1))
            interp = NearestNDInterpolator(np.transpose(mask), annotation[mask])
            filled_missing_annotations = interp(*np.indices(annotation.shape))
            self.annotations[image_path] = filled_missing_annotations

    def _check_annotations(self):
        """
        Checks that the annotations are correctly set up and that
        no unannotated minitiles exist.
        """
        for image_path, annotation in self.annotations.items():
            unannotated_minitiles = np.where(annotation < 0, 1, 0)
            unannotated_count = np.count_nonzero(unannotated_minitiles)
            if unannotated_count > 0:
                raise RuntimeError(f'Invalid annotation file for image: {image_path}. '
                                   f'Missing annotations: {unannotated_count}')

    def _make_dataset(self):
        """
        Prepares tf.data.Dataset.
        """
        # Keep all images with .annot.npy files in memory

        # Randomly select tiles from images and their corresponding annotations
        # until a full batch

        # Perform image augmentation
        ...


if __name__ == "__main__":
    dataset = TrainDataset('datasets/opssat/raw')
