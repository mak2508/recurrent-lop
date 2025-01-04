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
from src.utils import create_binary_task, plot_results, load_model, load_algo
from src.config import Config

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train Binary Tasks for Language Classification')
parser.add_argument('--config', type=str, required=True, help='Path to config file')
args = parser.parse_args()

# Load config file
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# Initialize config from dict
config = Config.from_dict(config)

# Set random seed for reproducibility
torch.manual_seed(42)

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

logging.info("Loading full dataset...")
full_dataset = load_full_dataset(dataset_file, config)
logging.info("Full dataset loaded successfully.")

# Prepare language pairs
logging.info("Preparing binary tasks...")
lang_pairs = list(itertools.permutations(range(len(config.languages)), 2))
random.shuffle(lang_pairs)

# Dynamically handle task count
task_batches = []
while len(task_batches) * len(lang_pairs) < config.num_tasks:
    random.shuffle(lang_pairs)
    task_batches.extend(lang_pairs)

# Limit total tasks to `config.num_tasks`
task_batches = task_batches[:config.num_tasks]

# Load model and algorithm
model = load_model(config)
algo = load_algo(model, config)

all_train_losses = []
all_test_accuracies = []

# Train on binary tasks
logging.info("Starting training...")
for task_idx, (lang1, lang2) in enumerate(task_batches):
    logging.info(f"Task {task_idx + 1}/{config.num_tasks}: Comparing {config.languages[lang1]} vs {config.languages[lang2]}")

    # Filter out samples for the current binary task
    binary_task = create_binary_task(full_dataset, (lang1, lang2))

    # Split into train and test datasets
    train_size = int(0.8 * len(binary_task))
    test_size = len(binary_task) - train_size
    train_dataset, test_dataset = random_split(binary_task, [train_size, test_size])

    # Train the model
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

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"output/{timestamp}/"
os.makedirs(output_dir, exist_ok=True)

np.save(f"{output_dir}/train_losses.npy", all_train_losses)
np.save(f"{output_dir}/test_accuracies.npy", all_test_accuracies)

# Plot results
plot_results(all_train_losses, all_test_accuracies, config.num_tasks, output_dir)
logging.info("Training complete.")
