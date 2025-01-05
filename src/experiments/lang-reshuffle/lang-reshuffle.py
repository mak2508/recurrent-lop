import torch
import numpy as np
import os
import sys
import argparse
import yaml
import logging
from datetime import datetime
from torch.utils.data import Dataset
import pandas as pd
import subprocess

from transformers import AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.basicConfig(level=logging.INFO)

# Get the absolute path to the parent directory of 'src'
project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..'))
sys.path.append(project_root)

from src.train import train_model
from src.utils import plot_results, shuffle_labels, load_model, load_algo
from src.config import Config

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train Language Classifier with Label Reshuffling')
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
        ["wget", "--no-check-certificate", "-O", dataset_file, "https://downloads.tatoeba.org/exports/sentences.csv"],
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

def load_language_data(dataset_file, config):
    logging.info("Loading and preprocessing language dataset...")
    df = pd.read_csv(dataset_file, sep='\t', header=None, quoting=3, names=['id', 'lang', 'sentence'])

    df = df.dropna(subset=['sentence'])
    df = df[df['lang'].isin(config.languages)]
    df = df[df['sentence'].str.strip() != '']

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    config.input_size = tokenizer.vocab_size  # Dynamically update input size
    logging.info(f"Tokenizer initialized with vocab_size: {tokenizer.vocab_size}")
    logging.info(f"Updated model input_size to match tokenizer vocab size: {config.input_size}")

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

    # Tokenize and pad sentences
    encoded_inputs = tokenizer(
        list(X_data),
        max_length=config.max_length,  # Reference correct attribute
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    X_data_padded = encoded_inputs["input_ids"]

    Y_data = torch.tensor(Y_data, dtype=torch.long)

    train_size = config.train_sentences_per_class * len(config.languages)
    X_train, Y_train = X_data_padded[:train_size], Y_data[:train_size]
    X_test, Y_test = X_data_padded[train_size:], Y_data[train_size:]

    train_dataset = LanguageDataset(X_train, Y_train)
    test_dataset = LanguageDataset(X_test, Y_test)
    return train_dataset, test_dataset, config.input_size


# Load data
train_dataset, test_dataset, vocab_size = load_language_data(dataset_file, config)
logging.info("Data preparation complete.")

# Load model and algorithm
model = load_model(config)
algo = load_algo(model, config)

all_train_losses = []
all_test_accuracies = []

# Repeat training with label reshuffling
for run in range(config.num_shuffles):
    logging.info(f"\nStarting Run {run + 1}/{config.num_shuffles}")
    
    # Shuffle the labels
    shuffled_train, label_mapping = shuffle_labels(train_dataset, None, config.num_classes)
    shuffled_test, _ = shuffle_labels(test_dataset, label_mapping, config.num_classes)

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

# Ensure output directory exists
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f'output/{timestamp}_{config.exp_desc}/'

os.makedirs(output_dir, exist_ok=True)
logging.debug(f"Directory '{output_dir}' was created.")

# Save training losses and test accuracies
np.save(f'{output_dir}/all_train_losses.npy', np.array(all_train_losses))
np.save(f'{output_dir}/all_test_accuracies.npy', np.array(all_test_accuracies))
logging.debug(f"Saved results to {output_dir}")

# Plot results
plot_results(all_train_losses, all_test_accuracies, config.num_shuffles, output_dir, config.exp_desc)