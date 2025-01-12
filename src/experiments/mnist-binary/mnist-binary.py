import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

import sys
import os
import argparse
import yaml
import logging
import random
import itertools
from datetime import datetime

logging.basicConfig(level=logging.INFO)

# Get the absolute path to the parent directory of 'src'
project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..'))
sys.path.append(project_root)

from src.train import train_model
from src.utils import create_binary_task, plot_results, plot_comparison, load_model, load_algo
from src.config import Config

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train MNIST binary tasks')
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

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False, 
                                          transform=transform)

if args.compare:
    # Comparison mode: Handle multiple configs
    results = {}
    exp_descs = []  # List to store experiment descriptions for plotting

    # Create a single output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'output/{timestamp}_comparison/'
    os.makedirs(output_dir, exist_ok=True)

    # Pre-generate binary tasks
    with open(args.config[0], 'r') as f:
        config_dict = yaml.safe_load(f)
        base_config = Config.from_dict(config_dict)

    binary_tasks = [
        (i, j) for i in range(base_config.num_classes) for j in range(i + 1, base_config.num_classes)
    ]  # Generate binary tasks
    binary_tasks = [binary_tasks[i] for i in torch.randperm(len(binary_tasks))]
    binary_tasks = binary_tasks[:base_config.num_tasks]

    # Process each configuration
    for config_path in args.config:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            config = Config.from_dict(config_dict)

        exp_descs.append(config.exp_desc)  # Collect experiment descriptions

        model = load_model(config)
        algo = load_algo(model, config)

        all_train_losses = []
        all_test_accuracies = []

        # Train on pre-generated binary tasks
        for task_idx, (class1, class2) in enumerate(binary_tasks):
            logging.info(f"\nStarting Binary Task {task_idx + 1}/{base_config.num_tasks} for {config.exp_desc}: Class {class1} vs Class {class2}")
            
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

        # Save results for this configuration
        final_accuracies_path = f'{output_dir}/{config.exp_desc}_final_accuracies.npy'
        np.save(final_accuracies_path, np.array([acc[-1] for acc in all_test_accuracies]))
        logging.info(f"Saved final accuracies for {config.exp_desc} to {final_accuracies_path}")

        results[config.exp_desc] = [acc[-1] for acc in all_test_accuracies]

    # Generate comparison plot
    plot_comparison(results, exp_descs, output_dir)

else:
    # Single config mode: Original behavior
    with open(args.config[0], 'r') as f:
        config_dict = yaml.safe_load(f)
        config = Config.from_dict(config_dict)

    model = load_model(config)
    algo = load_algo(model, config)

    random.seed(42)
    all_combinations = list(itertools.product(list(range(config.num_classes)), repeat=2))
    binary_tasks = [random.choice(all_combinations) for _ in range(config.num_tasks)]

    all_train_losses = []
    all_test_accuracies = []

    for task_idx, (class1, class2) in enumerate(binary_tasks):

        logging.info(f"\nStarting Binary Task {task_idx + 1}/{config.num_tasks}: Class {class1} vs Class {class2}")
        
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

    # Create an output directory for this config
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'output/{timestamp}_{config.exp_desc}/'
    os.makedirs(output_dir, exist_ok=True)

    # Save results
    np.save(f'{output_dir}/all_train_losses.npy', np.array(all_train_losses))
    np.save(f'{output_dir}/all_test_accuracies.npy', np.array(all_test_accuracies))

    plot_results(all_train_losses, all_test_accuracies, config.num_tasks, output_dir, config.exp_desc)
    logging.info(f"Results saved to {output_dir}")