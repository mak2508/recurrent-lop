import torch
from torch.utils.data import DataLoader
from typing import List, Tuple
from src.algos import AlgoType

import logging
# Get the root logger but don't modify its level
logger = logging.getLogger()

def train_model(
    algo: AlgoType,
    train_data: torch.utils.data.Dataset,
    test_data: torch.utils.data.Dataset,
    num_epochs: int,
    device: torch.device,
    batch_size: int,
) -> Tuple[List[float], List[float]]:
    model = algo.net
    
    # Training loop
    train_losses = []
    test_accuracies = []

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        model.train()

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Use Backprop's learn method
            loss, _ = algo.learn(images, labels)
            train_losses.append(loss.item())

        # Test accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for test_images, test_labels in test_loader:
                test_images = test_images.to(device)
                test_labels = test_labels.to(device)
                test_outputs = model(test_images)
                _, predicted = torch.max(test_outputs.data, 1)
                total += test_labels.size(0)
                correct += (predicted == test_labels).sum().item()

        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)

        logging.info(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.2f}%"
        )
        model.train()

    return train_losses, test_accuracies
