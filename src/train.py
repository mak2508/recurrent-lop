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
    train_accuracies = []
    test_accuracies = []

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        model.train()

        # Track training accuracy
        correct_train = 0
        total_train = 0
        epoch_loss = 0  # To accumulate loss for averaging per epoch

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Use Backprop's learn method
            loss, _ = algo.learn(images, labels)
            epoch_loss += loss.item()

            # Compute training accuracy
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Average training loss
        train_losses.append(epoch_loss / len(train_loader))

        # Calculate training accuracy
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)

        # Test accuracy
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for test_images, test_labels in test_loader:
                test_images = test_images.to(device)
                test_labels = test_labels.to(device)
                test_outputs = model(test_images)
                _, predicted = torch.max(test_outputs.data, 1)
                total_test += test_labels.size(0)
                correct_test += (predicted == test_labels).sum().item()

        test_accuracy = 100 * correct_test / total_test
        test_accuracies.append(test_accuracy)

        logging.info(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}, "
            f"Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%"
        )
        model.train()

    return train_losses, train_accuracies, test_accuracies
