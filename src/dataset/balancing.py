from typing import List

import numpy as np


def compute_class_weight(counts: List[int]):
    counts_np = np.array(counts)
    counts_sum = np.sum(counts_np)
    num_classes = np.shape(counts_np)[0]
    class_weight = counts_sum / (num_classes * counts_np)
    return class_weight
