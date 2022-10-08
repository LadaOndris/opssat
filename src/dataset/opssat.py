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

    def __init__(self, dataset_path: str, num_classes: int, minitile_size: int, batch_size: int):
        self.dataset_path = dataset_path
        self.num_classes = num_classes
        self.minitile_size = minitile_size
        self.batch_size = batch_size

        self.annotations = self._load_annotations()
        self._fill_unannotated_minitiles()
        self._check_annotations()

        self.image_file_paths = tf.convert_to_tensor([key for key in self.annotations.keys()])
        self.image_annotations = tf.convert_to_tensor([value for value in self.annotations.values()])
        self._check_all_classes_are_present()
        self.per_class_annotations = self._separate_annotations_based_on_label()

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

    def _separate_annotations_based_on_label(self):
        # [image][class][indices]
        separated_annotations = []
        # For every image
        for i in range(tf.shape(self.image_annotations)[0]):
            per_class_annotations = []
            # Check every classes and append indices to array
            for class_idx in range(self.num_classes):
                class_annotations = tf.where(self.image_annotations[i] == class_idx)
                per_class_annotations.append(class_annotations)
            separated_annotations.append(per_class_annotations)
        return separated_annotations

    def _build_iterator(self):
        """
        Prepares tf.data.Dataset.
        """
        datasets_for_each_class = []

        for class_idx in range(self.num_classes):
            dataset = tf.data.Dataset.from_tensors(self.image_file_paths)
            # dataset = dataset.map(self._get_filter_files_with_specific_class(class_idx))
            dataset = dataset.repeat()
            dataset = dataset.map(self._prepare_sample)
            dataset = dataset.filter(self._get_filter_class_predicate(class_idx))
            datasets_for_each_class.append(dataset)

        # Combine datasets together to get evenly represented classes
        choice_dataset = tf.data.Dataset.range(self.num_classes).repeat()
        combined_dataset = tf.data.Dataset.choose_from_datasets(datasets_for_each_class, choice_dataset)
        combined_dataset = combined_dataset.batch(self.batch_size)
        combined_dataset = combined_dataset.shuffle(self.batch_size)

        # Perform image augmentation

        return combined_dataset

    def _get_filter_files_with_specific_class(self, class_idx: int):
        def filter_with_specific_class(images_file_paths):
            mask = tf.zeros([tf.shape(images_file_paths)[0]], dtype=tf.int32)

            images_file_paths[]
            # for each image
            for i in range(len(self.per_class_annotations)):
                # for the specific class
                class_annotations = self.per_class_annotations[i][class_idx]
                if tf.size(class_annotations) > 0:
                    mask[i] = 1

            return images_file_paths[mask]
        return filter_with_specific_class

    def _prepare_sample(self, img_file_paths):
        # Select random image
        num_files = tf.shape(img_file_paths)[0]
        random_number = tf.random.uniform(shape=[1], maxval=num_files, dtype=tf.int32)[0]
        image_annotation = self.image_annotations[random_number]

        # Select random tile of the specific class
        self.per_class_annotations[random_number]

        # Select random tile 200x200 pixels
        tile_size = 200
        num_minitiles = int(tile_size // self.minitile_size)
        num_cols = tf.shape(image_annotation)[0]
        num_rows = tf.shape(image_annotation)[1]
        random_col = tf.random.uniform(shape=[1], maxval=num_cols - num_minitiles, dtype=tf.int32)[0]
        random_row = tf.random.uniform(shape=[1], maxval=num_rows - num_minitiles, dtype=tf.int32)[0]

        subimage_annotations = image_annotation[random_row:random_row + num_minitiles,
                               random_col:random_col + num_minitiles]

        # Create label of the majority of votes
        label = self._annotations_to_label(subimage_annotations)

        # Read image
        image_raw = tf.io.read_file(img_file_paths[random_number])
        image = tf.io.decode_png(image_raw, channels=3)

        # Extract tile
        tile_pixel_row = random_row * self.minitile_size
        tile_pixel_col = random_col * self.minitile_size
        tile = image[tile_pixel_row:tile_pixel_row + tile_size, tile_pixel_col:tile_pixel_col + tile_size, :]

        return tile, label

    def _get_filter_class_predicate(self, class_idx: int):

        def filter_class_predicate(tile, label) -> bool:
            return label == class_idx

        return filter_class_predicate

    def _annotations_to_label(self, annotations):
        y, idx, count = tf.unique_with_counts(tf.reshape(annotations, [-1]))
        river_label = 4
        if tf.reduce_any(y == river_label):
            return river_label
        most_votes_arg = tf.argmax(count)
        return y[most_votes_arg]


if __name__ == "__main__":
    dataset = TrainDataset('datasets/opssat/raw', num_classes=8, minitile_size=40, batch_size=32)
    for tile, annot in dataset.iterator:
        print(tile)
        print(annot)
        pass
