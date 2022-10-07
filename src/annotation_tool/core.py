from typing import Tuple


class Rectangle:

    def __init__(self, x, y, w, h) -> None:
        self.x = x
        self.y = y
        self.w = w
        self.h = h


def load_annotations():
    ...


def save_annotations(annotations) -> None:
    ...


def coordinates_to_rectangle(coordinates: Tuple[int, int]) -> Rectangle:
    ...
