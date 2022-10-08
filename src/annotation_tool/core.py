import os
from pathlib import Path
from typing import Tuple

import numpy as np


class StateStorage:

    def __init__(self, image_path: str, annotations_shape: Tuple[int, int], unannotated_flag=-1):
        self.annotations_shape = annotations_shape
        self.unannotated_flag = unannotated_flag

        self.annotation_path = self._image_to_annotation_path(image_path)
        self.annotations = self.load_annotations()

    def _image_to_annotation_path(self, image_path: str) -> str:
        path = Path(image_path)
        file_name_without_extension = path.stem
        file_path_without_filename = path.parent
        annotation_path_without_extension = os.path.join(file_path_without_filename, file_name_without_extension)
        annotation_path = f"{annotation_path_without_extension}.annot.npy"
        return annotation_path

    def get_annotations(self):
        return self.annotations

    def load_annotations(self) -> np.ndarray:
        if os.path.isfile(self.annotation_path):
            return np.load(self.annotation_path)
        else:
            return np.full(self.annotations_shape, fill_value=self.unannotated_flag, dtype=int)

    def save_annotations(self) -> None:
        np.save(self.annotation_path, self.annotations)

    def print_annotations_stats(self) -> None:
        annotated_mask = np.where(self.annotations >= 0, 1, 0)
        annotated_count = np.count_nonzero(annotated_mask)
        unannotated_count = self.annotations.size - annotated_count
        print(f"Annotated minitiles:\t{annotated_count}")
        print(f'Missing annotations:\t{unannotated_count}')

    def set_annotation(self, col: int, row: int, annotation_type: int) -> None:
        existing_annotation = self.annotations[row, col]
        if existing_annotation == annotation_type:
            self.annotations[row, col] = -1
        else:
            self.annotations[row, col] = annotation_type


def index_to_pixel(index, minitile_size) -> int:
    return index * minitile_size


def pixel_to_index(pixel, minitile_size) -> int:
    return int(pixel // minitile_size)
