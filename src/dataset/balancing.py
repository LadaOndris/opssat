from typing import Dict

import numpy as np


def compute_class_weight(counts: Dict[str, int]) -> Dict[str, int]:
    counts_np = np.array(list(counts.values()))
    counts_sum = np.sum(counts_np)
    num_classes = np.shape(counts_np)[0]
    class_weight = counts_sum / (num_classes * counts_np)
    class_weight_dict = {i: class_weight[i] for i in range(class_weight.shape[0])}
    return class_weight_dict
