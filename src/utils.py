import copy
import numpy as np
import torch
from typing import Optional, Tuple, List


def shuffle_labels(
    dataset: torch.utils.data.Dataset,
    new_label_mapping: Optional[np.ndarray] = None,
    num_classes: int = 10,
) -> Tuple[torch.utils.data.Dataset, np.ndarray]:
    """
    Creates a new dataset with shuffled labels.

    Args:
        dataset: The dataset to shuffle
        new_label_mapping: The new label mapping to use
        num_classes: The number of classes in the dataset

    Returns:
        A new dataset with shuffled labels and the new label mapping
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

def create_binary_task(
    dataset: torch.utils.data.Dataset,
    class_pair: Tuple[int, int],
    return_indices: bool = False,
) -> Tuple[torch.utils.data.Dataset, Optional[List[int]]]:
    """
    Creates a binary classification task from MNIST using two specified classes.
    
    Args:
        dataset: The MNIST dataset
        class_pair: Tuple of two class labels to use
        return_indices: If True, also return the indices used
    
    Returns:
        A subset of the dataset containing only the specified classes,
        with labels converted to 0 and 1
    """
    # Get indices for the two classes
    indices = []
    for idx, label in enumerate(dataset.targets):
        if label in class_pair:
            indices.append(idx)
    
    # Create a subset dataset
    subset = copy.deepcopy(dataset)
    subset.data = dataset.data[indices]
    subset.targets = dataset.targets[indices]
    
    # Convert labels to binary (0 and 1)
    subset.targets = torch.tensor([1 if label == class_pair[1] else 0 
                                 for label in subset.targets])
    
    if return_indices:
        return subset, indices
    return subset
