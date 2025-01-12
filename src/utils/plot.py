import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

def plot_results(
    all_train_losses: List[List[float]],
    all_test_accuracies: List[List[float]],
    num_tasks: int,
    save_path: Optional[str] = None,
    exp_desc: str = ""
) -> None:
    """
    Plot training losses and test accuracies across multiple runs.
    
    Args:
        all_train_losses (list): List of lists containing training losses for each run
        all_test_accuracies (list): List of lists containing test accuracies for each run
        num_tasks (int): Number of different label shuffles/runs
        save_path (str, optional): Base path to save the figures. If None, figures are not saved
        exp_desc (str): Experiment description
    """
    # Combine all losses and accuracies into single lists
    combined_losses = [loss for run_losses in all_train_losses for loss in run_losses]
    combined_accuracies = [acc for run_accuracies in all_test_accuracies for acc in run_accuracies]

    # Loss figure
    fig_loss = plt.figure(figsize=(15, 5))
    plt.plot(combined_losses)
    plt.title('Training Loss', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    for i in range(1, num_tasks):
        plt.axvline(x=i*len(combined_losses)//num_tasks, color='red', linestyle='--', alpha=0.5)
    if save_path:
        plt.savefig(f'{save_path}/loss.png')
    plt.close(fig_loss)

    # Accuracy figure
    fig_acc = plt.figure(figsize=(15, 5))
    plt.plot(combined_accuracies)
    plt.title('Test Accuracy', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    for i in range(1, num_tasks):
        plt.axvline(x=i*len(combined_accuracies)//num_tasks, color='red', linestyle='--', alpha=0.5)
    if save_path:
        plt.savefig(f'{save_path}/accuracy.png')
    plt.close(fig_acc)

    # Combined figure
    fig_combined, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    ax1.plot(combined_losses)
    ax1.set_title('Training Loss', fontsize=14)
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    for i in range(1, num_tasks):
        ax1.axvline(x=i*len(combined_losses)//num_tasks, color='red', linestyle='--', alpha=0.5)

    ax2.plot(combined_accuracies)
    ax2.set_title('Test Accuracy', fontsize=14)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    for i in range(1, num_tasks):
        ax2.axvline(x=i*len(combined_accuracies)//num_tasks, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(f'{save_path}/combined.png')
    #plt.show()


def plot_comparison(
    results: dict,
    config_files: list,
    save_path: Optional[str] = None
) -> None:
    """
    Plot line comparison of final accuracies across multiple configurations and save the output.

    Args:
        results (dict): Dictionary containing configuration names as keys and lists of final accuracies as values.
        config_files (list): List of config file paths used as input.
        save_path (str, optional): Path to save the figure. If None, the figure is not saved.
    """
    # Use base filenames as labels
    labels = [os.path.basename(config_file) for config_file in config_files]

    # Check if all labels exist in results
    for label in labels:
        if label not in results:
            raise KeyError(f"Config file '{label}' not found in results.")

    # Create output directory if save_path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    # Plot line comparison for each config
    plt.figure(figsize=(12, 7))
    for label in labels:
        plt.plot(
            results[label],
            label=label,
            marker='o',
            linestyle='-',
            linewidth=2,
            markersize=6  # Set the marker size (default is 6, you can adjust this value)
        )

    plt.title("Comparison of Final Accuracies Across Configurations", fontsize=16)
    plt.xlabel("Shuffle Index", fontsize=14)
    plt.ylabel("Final Accuracy (%)", fontsize=14)
    plt.legend(title="Configuration", fontsize=12)
    plt.grid(alpha=0.5)

    # Save the plot as an image
    if save_path:
        plot_file = os.path.join(save_path, "line_comparison_plot.png")
        plt.savefig(plot_file, format="png", dpi=300, bbox_inches="tight")

    plt.tight_layout()
    #plt.show()


def plot_comparison_full_length(
    numpy_files: List[str],
    config_files: List[str],
    type: str,
    comparison_name: str
):
    """
    Plots a comparison of accuracy/loss of last epoch during each experiments given the configuration files.

    Args:
        numpy_files (List[str]): A list of paths to NumPy files containing the data to be plotted.
        config_paths (List[str]): A list of paths to YAML configuration files providing additional settings.
        type (str): The type of comparison to perform (e.g., "line", "bar", "scatter").
    """
    
    print(type + ' plotting.')
    font_size = 25
    save_path = './output'
    labels = []

    plt.figure(figsize=(15, 5))
    plt.xlabel('Experiments', fontsize=font_size)

    if type == 'Accuracy':
        plt.title('Testing ' + type + ' Comparison', fontsize=font_size)
        plt.ylabel(type + ' (%)', fontsize=font_size)
    if type == 'Loss':
        plt.title('Training ' + type + ' Comparison', fontsize=font_size)
        plt.ylabel(type, fontsize=font_size)

    for numpy_path, config_path in zip(numpy_files, config_files):    

        if type == 'Accuracy':
            data_path = numpy_path + '/all_test_accuracies.npy'
        if type == 'Loss':
            data_path = numpy_path + '/all_train_losses.npy'
        
        try:  
            with open(data_path, 'r') as file:
                data = np.load(data_path)
                print('Found: ' + data_path)
        except FileNotFoundError:
            print(f"Error: The file '{data_path}' does not exist.")
        
        try:  
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                print('Found: ' + config_path)
        except FileNotFoundError:
            print(f"Error: The file '{config_path}' does not exist.")
        except yaml.YAMLError as e:
            print(f"Error: Failed to parse YAML file. {e}")

        labels.append(config['exp_desc'])
        plt.plot(data[:, -1])

    plt.legend(labels, fontsize=font_size, loc='lower left')
    plt.tight_layout()

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.savefig(save_path + '/' +comparison_name + '_' + type.lower() + '.png')
    print(type + ' saved to file: ' + save_path)
    print()

    #plt.show()