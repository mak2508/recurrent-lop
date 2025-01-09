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
from src.utils import shuffle_labels, plot_results, plot_comparison, load_model, load_algo
from src.config import Config

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train MNIST with label reshuffling')
parser.add_argument('--config', nargs='+', required=True, help='Path(s) to config file(s)')
parser.add_argument('--compare', action='store_true', help='Enable comparison mode for multiple configs')
args = parser.parse_args()

# Set random seed for reproducibility
torch.manual_seed(42)

# Load and preprocess MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)

if args.compare:
    # Comparison mode: Handle multiple configs
    results = {}
    reshuffling_runs = []
    exp_descs = []  # Store experiment descriptions for plotting

    # Pre-generate reshufflings for consistency across experiments
    for run in range(Config.default_num_tasks):  # Assume a default `num_tasks` in Config
        shuffled_train, label_mapping = shuffle_labels(train_dataset)
        shuffled_test, _ = shuffle_labels(test_dataset, label_mapping)
        reshuffling_runs.append((shuffled_train, shuffled_test, label_mapping))

    # Create a single output directory for all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'output/{timestamp}_comparison/'
    os.makedirs(output_dir, exist_ok=True)

    # Process each configuration
    for config_path in args.config:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            config = Config.from_dict(config_dict)

        exp_descs.append(config.exp_desc)

        model = load_model(config)
        algo = load_algo(model, config)

        final_accuracies = []

        # Train with pre-generated reshufflings
        for run, (shuffled_train, shuffled_test, label_mapping) in enumerate(reshuffling_runs):
            logging.info(f"\nStarting Run {run + 1}/{config.num_tasks} for {config.exp_desc}")
            
            # Train the model
            _, test_accuracies = train_model(
                algo=algo,
                train_data=shuffled_train,
                test_data=shuffled_test,
                num_epochs=config.num_epochs,
                device=config.device,
                batch_size=config.batch_size
            )

            final_accuracies.append(test_accuracies[-1])  # Final accuracy of the last epoch

        results[config.exp_desc] = final_accuracies

        # Save results for this config
        final_accuracies_path = f'{output_dir}/{config.exp_desc}_final_accuracies.npy'
        np.save(final_accuracies_path, np.array(final_accuracies))
        logging.info(f"Saved final accuracies for {config.exp_desc} to {final_accuracies_path}")

    # Generate comparison plot
    plot_comparison(results, exp_descs, output_dir)

else:
    # Single config mode: Original behavior
    with open(args.config[0], 'r') as f:
        config_dict = yaml.safe_load(f)
        config = Config.from_dict(config_dict)

    model = load_model(config)
    algo = load_algo(model, config)

    all_train_losses = []
    all_test_accuracies = []

    # Repeat training with different label shufflings
    for run in range(config.num_tasks):
        logging.info(f"\nStarting Run {run + 1}/{config.num_tasks}")
        
        # Shuffle the labels
        shuffled_train, label_mapping = shuffle_labels(train_dataset)
        shuffled_test, _ = shuffle_labels(test_dataset, label_mapping)
        
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

    # Create an output directory for this config
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'output/{timestamp}_{config.exp_desc}/'
    os.makedirs(output_dir, exist_ok=True)

    # Save results
    np.save(f'{output_dir}/all_train_losses.npy', np.array(all_train_losses))
    np.save(f'{output_dir}/all_test_accuracies.npy', np.array(all_test_accuracies))

    plot_results(all_train_losses, all_test_accuracies, config.num_tasks, output_dir, config.exp_desc)
    logging.info(f"Results saved to {output_dir}")