import copy
import numpy as np
import torch
from typing import Optional, Tuple


def shuffle_labels(
    dataset: torch.utils.data.Dataset,
    new_label_mapping: Optional[np.ndarray] = None,
    num_classes: int = 10,
) -> Tuple[torch.utils.data.Dataset, np.ndarray]:
    """
    Note that this ends up modifying the original dataset
    """
    if new_label_mapping is None:
        # Generate a random permutation of digits 0-9
        new_label_mapping = np.random.permutation(num_classes)

    # Create a deep copy of the dataset to avoid modifying original
    shuffled_dataset = copy.deepcopy(dataset)

    # Remap the labels according to the permutation
    shuffled_dataset.targets = torch.tensor(
        [new_label_mapping[label] for label in dataset.targets]
    )

    return shuffled_dataset, new_label_mapping
