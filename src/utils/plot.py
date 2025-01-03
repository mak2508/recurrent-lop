import matplotlib.pyplot as plt
from typing import List, Optional

def plot_results(
    all_train_losses: List[List[float]],
    all_test_accuracies: List[List[float]],
    num_shuffles: int,
    save_path: Optional[str] = None,
    model_name: str = ""
) -> None:
    """
    Plot training losses and test accuracies across multiple runs.
    
    Args:
        all_train_losses (list): List of lists containing training losses for each run
        all_test_accuracies (list): List of lists containing test accuracies for each run
        num_shuffles (int): Number of different label shuffles/runs
        save_path (str, optional): Base path to save the figures. If None, figures are not saved
        model_name (str): Name of the model for file naming
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
    for i in range(1, num_shuffles):
        plt.axvline(x=i*len(combined_losses)//num_shuffles, color='red', linestyle='--', alpha=0.5)
    if save_path:
        plt.savefig(f'{save_path}_{model_name}_loss.png')
    plt.close(fig_loss)

    # Accuracy figure
    fig_acc = plt.figure(figsize=(15, 5))
    plt.plot(combined_accuracies)
    plt.title('Test Accuracy', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    for i in range(1, num_shuffles):
        plt.axvline(x=i*len(combined_accuracies)//num_shuffles, color='red', linestyle='--', alpha=0.5)
    if save_path:
        plt.savefig(f'{save_path}_{model_name}_accuracy.png')
    plt.close(fig_acc)

    # Combined figure
    fig_combined, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    ax1.plot(combined_losses)
    ax1.set_title('Training Loss', fontsize=14)
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    for i in range(1, num_shuffles):
        ax1.axvline(x=i*len(combined_losses)//num_shuffles, color='red', linestyle='--', alpha=0.5)

    ax2.plot(combined_accuracies)
    ax2.set_title('Test Accuracy', fontsize=14)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    for i in range(1, num_shuffles):
        ax2.axvline(x=i*len(combined_accuracies)//num_shuffles, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(f'{save_path}_{model_name}_combined.png')
    plt.show()