import torch
import numpy as np
import os
import sys
import argparse
import yaml
import logging
from datetime import datetime
import pandas as pd
import subprocess
import random
import itertools
from torch.utils.data import Dataset, random_split

from transformers import AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.basicConfig(level=logging.INFO)

# Get the absolute path to the parent directory of 'src'
project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..'))
sys.path.append(project_root)

from src.train import train_model
from src.utils import create_binary_task, plot_results, plot_comparison, load_model, load_algo
from src.config import Config

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train Binary Tasks for Language Classification')
parser.add_argument('--config', nargs='+', required=True, help='Path(s) to config file(s)')
parser.add_argument('--compare', action='store_true', help='Enable comparison mode for multiple configs')
args = parser.parse_args()

# Download dataset if not present
dataset_file = 'tatoeba_sentences.csv'
if not os.path.exists(dataset_file):
    logging.info("Dataset not found. Downloading...")
    subprocess.run(
        ["wget", "-O", dataset_file, "https://downloads.tatoeba.org/exports/sentences.csv"],
        check=True
    )
    logging.info("Dataset downloaded successfully.")

# Define Dataset Class
class LanguageDataset(Dataset):
    def __init__(self, sentences, targets):
        self.sentences = sentences
        self.targets = targets

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.targets[idx]

# Load full dataset
def load_full_dataset(dataset_file, config):
    logging.info("Loading and preprocessing language dataset...")
    df = pd.read_csv(dataset_file, sep='\t', header=None, quoting=3, names=['id', 'lang', 'sentence'])

    df = df.dropna(subset=['sentence'])
    df = df[df['lang'].isin(config.languages)]
    df = df[df['sentence'].str.strip() != '']

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    config.input_size = tokenizer.vocab_size
    logging.info(f"Tokenizer initialized with vocab_size: {tokenizer.vocab_size}")

    X_data, Y_data = [], []
    for idx, lang in enumerate(config.languages):
        lang_df = df[df['lang'] == lang].sample(n=config.sentences_per_class, random_state=42)
        if len(lang_df) < config.sentences_per_class:
            raise ValueError(f"Not enough data for language: {lang}")

        X_data.extend(lang_df['sentence'].tolist())
        Y_data.extend([idx] * config.sentences_per_class)

    combined = list(zip(X_data, Y_data))
    np.random.seed(42)
    np.random.shuffle(combined)
    X_data, Y_data = zip(*combined)

    encoded_inputs = tokenizer(
        list(X_data),
        max_length=config.max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    X_data_padded = encoded_inputs["input_ids"]
    Y_data = torch.tensor(Y_data, dtype=torch.long)

    return LanguageDataset(X_data_padded, Y_data)

if args.compare:
    # Comparison mode: Handle multiple configs
    results = {}
    exp_descs = []  # List to store experiment descriptions for plotting

    # Create a single output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/{timestamp}_comparison/"
    os.makedirs(output_dir, exist_ok=True)

    config_path = args.config[0]
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        config = Config.from_dict(config_dict)

    # Prepare language pairs
    lang_pairs = list(itertools.permutations(range(len(config.languages)), 2))
    random.shuffle(lang_pairs)

    # Dynamically handle task count
    task_batches = []
    while len(task_batches) * len(lang_pairs) < config.num_tasks:
        random.shuffle(lang_pairs)
        task_batches.extend(lang_pairs)

    # Limit total tasks to `config.num_tasks`
    task_batches = task_batches[:config.num_tasks]

    for config_path in args.config:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            config = Config.from_dict(config_dict)

        exp_descs.append(config.exp_desc)  # Collect experiment descriptions
        config_name = os.path.basename(config_path)
        logging.info(f"Processing config: {config_name}")

        full_dataset = load_full_dataset(dataset_file, config)

        # Load model and algorithm
        model = load_model(config)
        algo = load_algo(model, config)

        final_accuracies = []

        # Train on binary tasks
        for task_idx, (lang1, lang2) in enumerate(task_batches):
            logging.info(f"\nStarting Task {task_idx + 1}/{config.num_tasks} for {config.exp_desc}: "
                         f"Comparing {config.languages[lang1]} vs {config.languages[lang2]}")
            binary_task = create_binary_task(full_dataset, (lang1, lang2))
            train_size = int(0.8 * len(binary_task))
            test_size = len(binary_task) - train_size
            train_dataset, test_dataset = random_split(binary_task, [train_size, test_size])

            _, test_accuracies = train_model(
                algo=algo,
                train_data=train_dataset,
                test_data=test_dataset,
                num_epochs=config.num_epochs,
                device=device,
                batch_size=config.batch_size,
            )

            final_accuracies.append(test_accuracies[-1])  # Final epoch accuracy

        results[config.exp_desc] = final_accuracies  # Use exp_desc as the key

        # Save final accuracies for this configuration in the unified directory
        final_accuracies_path = f"{output_dir}/{config.exp_desc}_final_accuracies.npy"
        np.save(final_accuracies_path, np.array(final_accuracies))
        logging.info(f"Saved final accuracies for {config.exp_desc} to {final_accuracies_path}")

    # Generate comparison plot
    plot_comparison(results, exp_descs, output_dir)
else:
    # Single config mode: Original behavior
    with open(args.config[0], 'r') as f:
        config = yaml.safe_load(f)
    config = Config.from_dict(config)

    full_dataset = load_full_dataset(dataset_file, config)

    # Prepare language pairs
    lang_pairs = list(itertools.permutations(range(len(config.languages)), 2))
    random.shuffle(lang_pairs)

    task_batches = []
    while len(task_batches) * len(lang_pairs) < config.num_tasks:
        random.shuffle(lang_pairs)
        task_batches.extend(lang_pairs)

    task_batches = task_batches[:config.num_tasks]

    model = load_model(config)
    algo = load_algo(model, config)

    all_train_losses = []
    all_test_accuracies = []

    for task_idx, (lang1, lang2) in enumerate(task_batches):
        logging.info(f"\nStarting Task {task_idx + 1}/{config.num_tasks} for {config.exp_desc}: "
                     f"Comparing {config.languages[lang1]} vs {config.languages[lang2]}")
        binary_task = create_binary_task(full_dataset, (lang1, lang2))
        train_size = int(0.8 * len(binary_task))
        test_size = len(binary_task) - train_size
        train_dataset, test_dataset = random_split(binary_task, [train_size, test_size])

        train_losses, test_accuracies = train_model(
            algo=algo,
            train_data=train_dataset,
            test_data=test_dataset,
            num_epochs=config.num_epochs,
            device=device,
            batch_size=config.batch_size,
        )

        all_train_losses.append(train_losses)
        all_test_accuracies.append(test_accuracies)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/{timestamp}/"
    os.makedirs(output_dir, exist_ok=True)

    np.save(f"{output_dir}/train_losses.npy", all_train_losses)
    np.save(f"{output_dir}/test_accuracies.npy", all_test_accuracies)

    plot_results(all_train_losses, all_test_accuracies, config.num_tasks, output_dir)
    logging.info("Training complete.")