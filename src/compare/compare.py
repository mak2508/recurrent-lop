import os
import sys
import yaml
import argparse

project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
sys.path.append(project_root)

from src.utils import plot_comparison_full_length

def compare_configs(
     config_path: str   
):
    """
    Reads the given YAML config file and plots comparison images.

    Args:
        config_path (str): Path to the YAML configuration file.
    """

    try:
        with open(config_path, 'r') as file:

            config = yaml.safe_load(file)
            config_name = config_path.split('/')[-1].split('.')[0]
            
            print()
            print('Configuration ' + config_name + ' loaded.')
            print()

    except FileNotFoundError:
        print(f"Error: The file '{config_path}' does not exist.")
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML file. {e}")

    if config['accuracy']:
        if 'labels' in config:
            plot_comparison_full_length(config['numpy_files'], config['config_files'], 'Accuracy', config_name, config['labels'])
        else:
            plot_comparison_full_length(config['numpy_files'], config['config_files'], 'Accuracy', config_name)
    
    if config['loss']:
        if 'labels' in config:
            plot_comparison_full_length(config['numpy_files'], config['config_files'], 'Loss', config_name, config['labels'])
        else:
            plot_comparison_full_length(config['numpy_files'], config['config_files'], 'Loss', config_name)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparison configuration file.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML config file.")
    args = parser.parse_args()
    compare_configs(args.config)
