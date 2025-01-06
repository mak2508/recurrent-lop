import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

import sys
import os
import argparse
import yaml
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)

# Get the absolute path to the parent directory of 'src'
project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..'))
sys.path.append(project_root)

from src.train import train_model
from src.utils import create_binary_task, plot_results, load_model, load_algo
from src.config import Config

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train MNIST birary tasks')
parser.add_argument('--config', type=str, required=True, help='Path to config file')
args = parser.parse_args()

# Load config file
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# Initialize config from dict
config = Config.from_dict(config)

# Set random seed for reproducibility
torch.manual_seed(42)

# Load and preprocess MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', 
                                         train=True,
                                         transform=transform,
                                         download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                        train=False, 
                                        transform=transform)


model = load_model(config)

algo = load_algo(model, config)

# Define sequence of binary tasks
binary_tasks = [
    (i, j) for i in range(config.num_classes) for j in range(i+1, config.num_classes)
]  # Generate all possible pairs of digits (45 pairs total)
binary_tasks = [binary_tasks[i] for i in torch.randperm(len(binary_tasks))]  # Randomly shuffle the order of tasks
binary_tasks = binary_tasks[:config.num_tasks]

all_train_losses = []
all_test_accuracies = []
task_boundaries = []

logging.info("Starting training...")
for task_idx, (class1, class2) in enumerate(binary_tasks):
    logging.debug(f"\nStarting Binary Task {task_idx + 1}: Class {class1} vs Class {class2}")
    
    # Create binary datasets
    binary_train = create_binary_task(train_dataset, (class1, class2))
    binary_test = create_binary_task(test_dataset, (class1, class2))
    
    # Train the model on this binary task
    train_losses, test_accuracies = train_model(
        algo=algo,
        train_data=binary_train,
        test_data=binary_test,
        num_epochs=config.num_epochs,
        device=config.device,
        batch_size=config.batch_size
    )

    all_train_losses.append(train_losses)
    all_test_accuracies.append(test_accuracies)
    task_boundaries.append(len(all_train_losses))
    

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f'output/{timestamp}_{config.exp_desc}/'
os.makedirs(output_dir, exist_ok=True)

# Save training losses and test accuracies to numpy files
np.save(f'{output_dir}/all_train_losses.npy', np.array(all_train_losses))
np.save(f'{output_dir}/all_test_accuracies.npy', np.array(all_test_accuracies))
logging.debug(f"Saved training losses and test accuracies to {output_dir}")

plot_results(all_train_losses, all_test_accuracies, config.num_tasks, output_dir, config.exp_desc)