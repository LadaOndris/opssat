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

    def __init__(self, dataset_path: str, num_classes: int, minitile_size: int, batch_size: int,
                 tile_size: int = 200, min_minitiles: int = 2, max_minitiles: int = 10, max_minitile_ration: int = 3):
        self.dataset_path = dataset_path
        self.num_classes = num_classes
        self.minitile_size = minitile_size
        self.batch_size = batch_size
        self.tile_size = tile_size  # Size in pixels

        self.annotations = self._load_annotations()
        self._fill_unannotated_minitiles()
        self._check_annotations()

        self.image_file_paths = tf.convert_to_tensor([key for key in self.annotations.keys()])
        self.image_annotations = tf.convert_to_tensor([value for value in self.annotations.values()])
        self._check_all_classes_are_present()
        self.class_weights = self._compute_class_weights()

        self.allowed_tile_sizes = self._prepare_allowed_tilesizes(min_minitiles, max_minitiles,
                                                                  max_allowed_ratio=max_minitile_ration)
        self.iterator = self._build_iterator()

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

    def _check_all_classes_are_present(self):
        for class_idx in range(self.num_classes):
            class_exists = tf.reduce_any(self.image_annotations == class_idx)
            if not class_exists:
                raise RuntimeError(f"Class {class_idx} is not present in training data.")

    def _compute_class_weights(self):
        class_counts = np.empty(shape=(self.num_classes,), dtype=float)
        for class_idx in range(self.num_classes):
            class_mask = tf.cast(self.image_annotations == class_idx, dtype=tf.int32)
            class_count = tf.reduce_sum(class_mask)
            class_counts[class_idx] = class_count

        normalized_counts = np.max(class_counts) / class_counts
        class_weights = {i: normalized_counts[i] for i in range(normalized_counts.shape[0])}
        return class_weights

    def _build_iterator(self):
        """
        Prepares tf.data.Dataset.
        """
        dataset = tf.data.Dataset.from_tensors(self.image_file_paths)
        dataset = dataset.repeat()
        dataset = dataset.map(self._prepare_sample)
        dataset = dataset.map(self._augment_image)
        dataset = dataset.batch(self.batch_size)
        return dataset

    def _prepare_sample(self, img_file_paths):
        # Select random image
        num_files = tf.shape(img_file_paths)[0]
        random_number = tf.random.uniform(shape=[1], maxval=num_files, dtype=tf.int32)[0]
        image_annotation = self.image_annotations[random_number]

        # Select random tile size
        tile_size_idx = tf.random.uniform(shape=[1], maxval=tf.shape(self.allowed_tile_sizes)[0], dtype=tf.int32)[0]
        num_minitiles = self.allowed_tile_sizes[tile_size_idx]

        # Select random tile 200x200 pixels
        num_cols = tf.shape(image_annotation)[0]
        num_rows = tf.shape(image_annotation)[1]
        random_col = tf.random.uniform(shape=[1], maxval=num_cols - num_minitiles[0], dtype=tf.int32)[0]
        random_row = tf.random.uniform(shape=[1], maxval=num_rows - num_minitiles[1], dtype=tf.int32)[0]

        subimage_annotations = image_annotation[random_row:random_row + num_minitiles[1],
                               random_col:random_col + num_minitiles[0]]

        # Create label of the majority of votes
        label = self._annotations_to_label(subimage_annotations)

        # Read image
        image_raw = tf.io.read_file(img_file_paths[random_number])
        image = tf.io.decode_png(image_raw, channels=3)

        # Extract tile
        start_tile_pixel_row = random_row * self.minitile_size
        start_tile_pixel_col = random_col * self.minitile_size
        end_tile_pixel = num_minitiles * self.minitile_size

        tile = image[start_tile_pixel_row:start_tile_pixel_row + end_tile_pixel[1],
               start_tile_pixel_col:start_tile_pixel_col + end_tile_pixel[0], :]
        resized_tile = tf.image.resize(tile, [self.tile_size, self.tile_size])
        resized_tile = tf.cast(resized_tile, dtype=tf.int32)

        return resized_tile, label

    def _augment_image(self, tile, annot):
        # tile = tile_annot_pair[0]
        # annot = tile_annot_pair[1]

        flipped = tf.image.random_flip_left_right(tile)
        flipped_twice = tf.image.random_flip_up_down(flipped)
        return flipped_twice, annot

    def _prepare_allowed_tilesizes(self, min_minitiles: int, max_minitiles: int,
                                   max_allowed_ratio: int) -> tf.TensorArray:
        # Prepare possible minitile sizes
        # select 2x2 to 10x10 with aspecet ration max 3 (allowed 2x6, 3x9, but not 3x10)
        assert (min_minitiles > 0)
        assert (max_minitiles >= min_minitiles)

        a = np.arange(min_minitiles, max_minitiles + 1)
        x, y = np.meshgrid(a, a)
        xy_matrix = np.stack([x, y], axis=-1)
        xy = xy_matrix.reshape([-1, 2])

        xy_ratio_mask_one = (xy[:, 0] / xy[:, 1]) <= max_allowed_ratio
        xy_ratio_mask_two = (xy[:, 1] / xy[:, 0]) <= max_allowed_ratio

        allowed_tilesizes = xy[xy_ratio_mask_one & xy_ratio_mask_two]
        return tf.convert_to_tensor(allowed_tilesizes, dtype=tf.int32)

    def _annotations_to_label(self, annotations):
        y, idx, count = tf.unique_with_counts(tf.reshape(annotations, [-1]))
        river_label = 4
        if tf.reduce_any(y == river_label):
            return river_label
        most_votes_arg = tf.argmax(count)
        return y[most_votes_arg]


if __name__ == "__main__":
    import matplotlib

    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    dataset = TrainDataset('datasets/opssat/raw', num_classes=8, minitile_size=40, batch_size=32)
    for tile_batch, annot_batch in dataset.iterator:
        tile_batch_np = tile_batch.numpy()
        for i in range(np.shape(tile_batch_np)[0]):
            plt.imshow(tile_batch_np[i])
            plt.show()
            pass
