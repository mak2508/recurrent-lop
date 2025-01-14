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

train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True,
                                           transform=transform,
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data/',
                                          train=False, 
                                          transform=transform)

if args.compare:
    # Comparison mode: Handle multiple configs with consistent reshuffling
    exp_descs = []
    reshuffling_runs = []
    results = {}  # Add results dictionary to store final accuracies for each configuration

    # Load the first config to determine reshuffling parameters
    with open(args.config[0], 'r') as f:
        config_dict = yaml.safe_load(f)
        base_config = Config.from_dict(config_dict)

    # Pre-generate all reshufflings
    for run in range(base_config.num_tasks):
        shuffled_train, label_mapping = shuffle_labels(train_dataset)
        shuffled_test, _ = shuffle_labels(test_dataset, label_mapping)
        reshuffling_runs.append((shuffled_train, shuffled_test))

    # Create a single output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'output/{timestamp}_comparison/'
    os.makedirs(output_dir, exist_ok=True)

    # Apply reshuffling to all configurations
    for config_path in args.config:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            config = Config.from_dict(config_dict)

        exp_descs.append(config.exp_desc)
        exp_output_dir = os.path.join(output_dir, config.exp_desc)
        os.makedirs(exp_output_dir, exist_ok=True)  # Create a folder for this configuration

        model = load_model(config)
        algo = load_algo(model, config)

        run_train_losses = []
        run_test_accuracies = []
        final_accuracies = []  # Track final accuracies for the current config

        for run, (shuffled_train, shuffled_test) in enumerate(reshuffling_runs):
            logging.info(f"\nStarting Run {run + 1}/{base_config.num_tasks} for {config.exp_desc}")

            train_losses, test_accuracies = train_model(
                algo=algo,
                train_data=shuffled_train,
                test_data=shuffled_test,
                num_epochs=config.num_epochs,
                device=config.device,
                batch_size=config.batch_size
            )
            run_train_losses.append(train_losses)
            run_test_accuracies.append(test_accuracies)
            final_accuracies.append(test_accuracies[-1])  # Append the final accuracy for this run

        results[config.exp_desc] = final_accuracies  # Store final accuracies in the results dictionary

        # Save results for this configuration
        np.save(os.path.join(exp_output_dir, 'train_losses.npy'), np.array(run_train_losses))
        np.save(os.path.join(exp_output_dir, 'test_accuracies.npy'), np.array(run_test_accuracies))
        logging.info(f"Results saved for {config.exp_desc} in {exp_output_dir}")

    # Generate comparison plot using results
    plot_comparison(results, exp_descs, output_dir)

else:
    # Single config mode: Original behavior with preloaded reshufflings
    with open(args.config[0], 'r') as f:
        config_dict = yaml.safe_load(f)

    config = Config.from_dict(config_dict)

    # Pre-generate all reshufflings
    reshuffling_runs = []
    for run in range(config.num_tasks):
        shuffled_train, label_mapping = shuffle_labels(train_dataset)
        shuffled_test, _ = shuffle_labels(test_dataset, label_mapping)
        reshuffling_runs.append((shuffled_train, shuffled_test))

    model = load_model(config)
    algo = load_algo(model, config)

    all_train_losses = []
    all_test_accuracies = []

    for run, (shuffled_train, shuffled_test) in enumerate(reshuffling_runs):
        logging.info(f"\nStarting Run {run + 1}/{config.num_tasks}")

        train_losses, test_accuracies = train_model(
            algo=algo,
            train_data=shuffled_train,
            test_data=shuffled_test,
            num_epochs=config.num_epochs,
            device=config.device,
            batch_size=config.batch_size
        )

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

    plot_results(all_train_losses, all_test_accuracies, config.num_tasks, output_dir, config.exp_desc)
