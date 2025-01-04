# Recurrent Loss of Plasticity (LOP)

This repository explores [Loss of Plasticity](https://www.nature.com/articles/s41586-024-07711-7) in Recurrent Neural Networks.

## Usage

To run a particular experiment, navigate to the experiment subfolder and run as follows:
```
python <exp-name>.py --config <config-path>
```

Make sure to have access to a GPU in order to run at a reasonable pace. One way to do this is to sign into https://jupyter.euler.hpc.ethz.ch with GPU enabled and simply run this in a terminal instance here.

The better way to do this is using srun/sbatch through ssh into euler (Instructions on this coming soon.)


## Folder Structure
Within the `src` folder, the code is broken up into the following folders.

- `experiments`: This contains a separate subfolder for each of the experiments we run. More details on each experiment are provided in correspoding `README` files in each of the subfolders.
- `nets`: This contains all the different network architectures we use in our experiments. This can easily be extended by adding another file here.
- `algos`: This folder contains different learning algorithms, such as backpropagation and continual backpropagation.
- `utils`: Here we have several useful functionalities that are reused throughout the repository.

## Experiment Structure
Each experiment is setup as a python script that reads a config file as input with the specifications of its experiments. Below is a sample config file for the `mnist-reshuffle` task:

```yaml
# Model Configuration
model:
  model_type: "MLP"  # Model type (MLP or LSTM)
  input_size: 784 # 28 * 28 for mnist images
  hidden_size: 25
  num_classes: 10
  dropout_rate: 0.0

# Training Configuration
training:
  algo: "BP" # BP or CBP (yet to be implemented)
  num_epochs: 2  # Set to 25 for full training
  batch_size: 6000
  learning_rate: 0.01
  num_shuffles: 3  # Number of times to repeat training with different label shufflings
  to_perturb: False

exp_desc: "mlp"  # Experiment description for output directory naming

```

In the above, we can easily customize various specs such as the model type, number of experiments, etc.