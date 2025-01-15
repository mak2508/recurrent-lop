import os
import numpy as np

def calculate_and_save_accuracy_degradation(root_dir):
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

                    # Ensure there are at least two experiments
                    if len(final_accuracies) < 2:
                        raise ValueError("Not enough experiments to calculate accuracy degradation.")

                    # Extract initial and final accuracy
                    initial_accuracy = float(final_accuracies[0])
                    final_accuracy = float(final_accuracies[-1])

                    # Calculate the accuracy degradation
                    accuracy_degradation = ((initial_accuracy - final_accuracy) / initial_accuracy) * 100

                    # Save the accuracy degradation as a .txt file
                    txt_path = os.path.join(dirpath, "accuracy_degradation.txt")
                    with open(txt_path, "w") as txt_file:
                        txt_file.write(f"Accuracy Degradation: {accuracy_degradation:.4f}%")

                    print(f"Processed: {npy_path}, Accuracy Degradation: {accuracy_degradation:.4f}%")
                except Exception as e:
                    print(f"Error processing {npy_path}: {e}")

if __name__ == "__main__":
    # Set the root directory to the directory of the script
    root_directory = os.path.dirname(os.path.abspath(__file__))
    calculate_and_save_accuracy_degradation(root_directory)
