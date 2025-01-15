gru_mitigations_config = "../compare/configs/compare_lang_reshuffle_gru_mitigations.yaml"
mlp_mitigations_config = "../compare/configs/compare_mnist_reshuffle_mitigations.yaml"


import os
import yaml

def get_numpy_paths(config_path):
    # Read the YAML config file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract the numpy file paths
    numpy_paths = config.get('numpy_files', [])
    
    # Convert relative paths to absolute paths
    abs_paths = []
    config_dir = os.path.dirname(os.path.dirname(os.path.abspath(config_path)))
    for path in numpy_paths:
        abs_path = os.path.abspath(os.path.join(config_dir, path))
        abs_paths.append(abs_path)
        
    return abs_paths

# Get paths from both config files
gru_paths = get_numpy_paths(gru_mitigations_config)
mlp_paths = get_numpy_paths(mlp_mitigations_config)

print("GRU paths: ", gru_paths)
print("MLP paths: ", mlp_paths)
print("--------------------------------")

def read_mean_accuracy(path):
    # Construct path to mean_accuracy.txt
    accuracy_file = os.path.join(path, "mean_accuracy.txt")
    
    try:
        with open(accuracy_file, 'r') as f:
            content = f.read()
            # Extract the float value from the text
            accuracy = float(content.split(":")[1].strip())
            return accuracy
    except Exception as e:
        print(f"Error reading accuracy from {accuracy_file}: {e}")
        return None

# Get base accuracies (first path in each list is the baseline)
gru_base_accuracy = read_mean_accuracy(gru_paths[0])
mlp_base_accuracy = read_mean_accuracy(mlp_paths[0])

print(f"GRU baseline accuracy: {gru_base_accuracy}")
print(f"MLP baseline accuracy: {mlp_base_accuracy}")


# Read accuracies for each mitigation technique
gru_l2_accuracy = read_mean_accuracy(gru_paths[1])
gru_perturb_accuracy = read_mean_accuracy(gru_paths[2]) 
gru_cbp_accuracy = read_mean_accuracy(gru_paths[3])

mlp_l2_accuracy = read_mean_accuracy(mlp_paths[1])
mlp_perturb_accuracy = read_mean_accuracy(mlp_paths[2])
mlp_cbp_accuracy = read_mean_accuracy(mlp_paths[3])

print("\nGRU mitigation accuracies:")
print(f"L2: {gru_l2_accuracy}")
print(f"Perturb: {gru_perturb_accuracy}")
print(f"CBP: {gru_cbp_accuracy}")

print("\nMLP mitigation accuracies:")
print(f"L2: {mlp_l2_accuracy}")
print(f"Perturb: {mlp_perturb_accuracy}") 
print(f"CBP: {mlp_cbp_accuracy}")


def calculate_iaa(mitigation_accuracy, base_accuracy):
    """
    Calculate the Improvement over Average Accuracy (IAA) score
    
    Args:
        mitigation_accuracy: Accuracy score for the mitigation technique
        base_accuracy: Baseline accuracy score to compare against
        
    Returns:
        float: IAA score as a percentage
    """
    return ((mitigation_accuracy - base_accuracy) / base_accuracy) * 100

# Calculate IAA for GRU mitigations
gru_l2_iaa = calculate_iaa(gru_l2_accuracy, gru_base_accuracy)
gru_perturb_iaa = calculate_iaa(gru_perturb_accuracy, gru_base_accuracy)
gru_cbp_iaa = calculate_iaa(gru_cbp_accuracy, gru_base_accuracy)

# Calculate IAA for MLP mitigations
mlp_l2_iaa = calculate_iaa(mlp_l2_accuracy, mlp_base_accuracy)
mlp_perturb_iaa = calculate_iaa(mlp_perturb_accuracy, mlp_base_accuracy)
mlp_cbp_iaa = calculate_iaa(mlp_cbp_accuracy, mlp_base_accuracy)

print("\nGRU IAA scores:")
print(f"L2: {gru_l2_iaa:.2f}%")
print(f"Perturb: {gru_perturb_iaa:.2f}%") 
print(f"CBP: {gru_cbp_iaa:.2f}%")

print("\nMLP IAA scores:")
print(f"L2: {mlp_l2_iaa:.2f}%")
print(f"Perturb: {mlp_perturb_iaa:.2f}%")
print(f"CBP: {mlp_cbp_iaa:.2f}%")


