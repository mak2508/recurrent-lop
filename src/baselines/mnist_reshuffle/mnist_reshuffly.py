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
project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
sys.path.append(project_root)

from src.algos import BP
from src.nets import MLP, LSTM
from src.train import train_model
from src.utils import shuffle_labels, plot_results, load_model, load_algo
from src.config import Config

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train MNIST with label reshuffling')
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

all_train_losses = []
all_test_accuracies = []

# Repeat training with different label shufflings
for run in range(config.num_shuffles):
    logging.info(f"\nStarting Run {run + 1}/{config.num_shuffles}")
    
    # Shuffle the labels
    shuffled_train, label_mapping = shuffle_labels(train_dataset)
    shuffled_test, _ = shuffle_labels(test_dataset, label_mapping)  # Use same mapping for test set
    
    # Print the label mapping for this run
    logging.debug("Label mapping for this run:")
    logging.debug("Original:  ", " ".join(str(i) for i in range(10)))
    logging.debug("Mapped to: ", " ".join(str(label_mapping[i]) for i in range(10)))
    
    # Train the model
    train_losses, test_accuracies = train_model(
        algo=algo,
        train_data=shuffled_train,
        test_data=shuffled_test,
        num_epochs=config.num_epochs,
        device=config.device,
        batch_size=config.batch_size
    )

    # Store results
    all_train_losses.append(train_losses)
    all_test_accuracies.append(test_accuracies)


# Ensure 'output' directory exists
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f'output/{timestamp}_{config.exp_desc}/'

os.makedirs(output_dir)
logging.debug(f"Directory '{output_dir}' was created.")

# Save training losses and test accuracies to numpy files
np.save(f'{output_dir}/all_train_losses.npy', np.array(all_train_losses))
np.save(f'{output_dir}/all_test_accuracies.npy', np.array(all_test_accuracies))
logging.debug(f"Saved training losses and test accuracies to {output_dir}")

plot_results(all_train_losses, all_test_accuracies, config.num_shuffles, output_dir, config.model_type)