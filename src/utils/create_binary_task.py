import torch
from typing import Optional, Tuple, List
from torch.utils.data import Dataset

def create_binary_task(
    dataset: Dataset,
    class_pair: Tuple[int, int],
    return_indices: bool = False,
) -> Tuple[Dataset, Optional[List[int]]]:
    """
    Creates a binary classification task from a dataset using two specified classes.

    Args:
        dataset: A PyTorch dataset with `sentences` and `targets` attributes.
        class_pair: Tuple of two class labels to filter on.
        return_indices: If True, also return the indices used.

    Returns:
        A dataset containing only the specified classes,
        with labels converted to 0 and 1.
    """
    # Unpack the class pair
    class_0, class_1 = class_pair

    # Filter samples belonging to the two classes
    indices = [idx for idx, label in enumerate(dataset.targets) if label == class_0 or label == class_1]

    if not indices:
        raise ValueError(f"No samples found for class pair {class_pair} in the dataset.")

    # Remap labels to binary
    class_to_binary = {class_0: 0, class_1: 1}
    filtered_sentences = [dataset.sentences[idx] for idx in indices]
    filtered_labels = [class_to_binary[dataset.targets[idx].item()] for idx in indices]

    # Define a new dataset for binary classification
    class BinaryDataset(Dataset):
        def __init__(self, sentences, targets):
            self.sentences = sentences
            self.targets = targets

        def __len__(self):
            return len(self.sentences)

        def __getitem__(self, idx):
            return self.sentences[idx], self.targets[idx]

    binary_dataset = BinaryDataset(filtered_sentences, filtered_labels)

    if return_indices:
        return binary_dataset, indices
    return binary_dataset
