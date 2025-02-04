import os
import numpy as np

def calculate_and_save_mean_accuracy(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == "all_test_accuracies.npy":
                npy_path = os.path.join(dirpath, filename)
                try:
                    # Load the numpy array
                    accuracies = np.load(npy_path)

                    # Ensure accuracies has the correct shape
                    if accuracies.ndim != 2:
                        raise ValueError(f"Unexpected array shape: {accuracies.shape}. Expected 2D array.")

                    # Extract only the last epoch for each experiment
                    final_accuracies = accuracies[:, -1]

                    # Calculate the mean of the last epoch accuracies
                    mean_accuracy = np.mean(final_accuracies)

                    # Save the mean accuracy as a .txt file
                    txt_path = os.path.join(dirpath, "mean_accuracy.txt")
                    with open(txt_path, "w") as txt_file:
                        txt_file.write(f"Mean Accuracy (Last Epoch): {mean_accuracy:.4f}")

                    print(f"Processed: {npy_path}, Mean Accuracy (Last Epoch): {mean_accuracy:.4f}")
                except Exception as e:
                    print(f"Error processing {npy_path}: {e}")

if __name__ == "__main__":
    # Set the root directory to the directory of the script
    root_directory = os.path.dirname(os.path.abspath(__file__))
    calculate_and_save_mean_accuracy(root_directory)
