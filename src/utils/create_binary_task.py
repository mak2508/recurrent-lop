import torch
from typing import Optional, Tuple, List
from torch.utils.data import Dataset, Subset

def create_binary_task(
    dataset: torch.utils.data.Dataset,
    class_pair: Tuple[int, int],
    return_indices: bool = False,
) -> Tuple[torch.utils.data.Dataset, Optional[List[int]]]:
    """
    Creates a binary classification task from a dataset using two specified classes.
    
    Args:
        dataset: A PyTorch dataset with a targets attribute or equivalent.
        class_pair: Tuple of two class labels to use.
        return_indices: If True, also return the indices used.
    
    Returns:
        A subset of the dataset containing only the specified classes,
        with labels converted to 0 and 1.
    """
    # Get indices for the two classes
    indices = [idx for idx, label in enumerate(dataset.targets) if label in class_pair]
    
    # Create a subset of the original dataset
    subset = Subset(dataset, indices)
    
    # Convert labels to binary (0 and 1)
    class_to_binary = {class_pair[0]: 0, class_pair[1]: 1}
    
    # Wrapper to override labels in the subset
    class BinaryDatasetWrapper(Dataset):
        def __init__(self, original_dataset, indices):
            self.dataset = original_dataset
            self.indices = indices
            self.targets = [class_to_binary[self.dataset.targets[idx]] for idx in indices]
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, i):
            image, _ = self.dataset[self.indices[i]]  # Ignore original label
            return image, self.targets[i]

    binary_dataset = BinaryDatasetWrapper(dataset, indices)
    
    if return_indices:
        return binary_dataset, indices
    return binary_dataset