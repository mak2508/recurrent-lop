# Recurrent Loss of Plasticity (LOP)

This repository explores [Loss of Plasticity](https://www.nature.com/articles/s41586-024-07711-7) in Recurrent Neural Networks. All experiments we propose have their correspoding output already generated, However, if you want to run them for yourself we will guide you thorugh the process!

## Report

You can find our project report writeup on the class OpenReview console [here](https://openreview.net/forum?id=0mXNceVrV2&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DETHz.ch%2F2024%2FCourse%2FDL_Reports_Early_Deadline%2FAuthors%23your-submissions)).

The project proposal is available [here](https://openreview.net/forum?id=HB0myU9MTK&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DETHz.ch%2F2024%2FCourse%2FDL_Early_Deadline%2FAuthors%23your-submissions)).

## Folder Structure
Within the `src` folder, the code is broken up into the following folders.

- `experiments`: This contains a separate subfolder for each of the experiments we run. More details on each experiment are provided in correspoding `README` files in each of the subfolders.
- `nets`: This contains all the different network architectures we use in our experiments. This can easily be extended by adding another file here.
- `algos`: This folder contains different learning algorithms, such as backpropagation and continual backpropagation.
- `utils`: Here we have several useful functionalities that are reused throughout the repository.

## Usage

### Setting up

As a start clone our repository:
```shell
# clone repo
git clone https://github.com/mak2508/recurrent-lop.git
cd recurrent-lop
```

Next create an environment and activate it (you may use any python version you like, this is what works for us):
```shell
# create env cuda XXX
conda create --name dl_env python=3.8 pip
conda activate dl_env
```

If you are using slurm based system you can load modules now. For ETH Euler cluster we used:
```shell
# load modules
module load stack/2024-06  gcc/12.2.0
module load cuda/12.1.1

# check whether they are loaded properly
gcc --version
nvcc --version
```

Download torch version for your cuda version:
```shell
# instal torch for cuda XXX ---> https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 #change here cuda versions
```

Now install any additional dependencies:
```shell
# install other dependencies
pip install -r requirements.txt
```

You are ready to go!

## Running

### Experiments

Make sure to have access to a GPU in order to run at a reasonable pace. One way to do this is to sign into https://jupyter.euler.hpc.ethz.ch with GPU enabled and simply run this in a terminal instance here. Feel free to change any configuration. The better way to do this is using srun/sbatch through ssh into ETH Euler cluster.

To run a particular experiment locally, navigate to the experiment subfolder and run as follows:

```shell
python <exp-name>.py --config <config-path>
```

or

```shell
./<exp-name>.sh
```

If you want to run multiple configurations with the same experiment rounds for the same task, you can use the `--compare` flag as follows:

```shell
python <exp-name>.py --compare --config <config-path-1> <config-path-2> ...
```

If you are using a remote machine, such as ETH Euler cluster, you should use our bash scripts too:

```shell
srun --time=8:00:00 --gpus=1 --gres=gpumem:8g --mem-per-cpu=16g <exp-name>.sh
```

or

```shell
sbatch --time=8:00:00 --gpus=1 --gres=gpumem:8g --mem-per-cpu=16g <exp-name>.sh
```

If you encounter a permission error, you can run:

```shell
chmod +x <exp-name>.sh
```

Feel free to change any configuration setting in srun/sbatch, however, these are enough to run the files in reasonable speed. Running our bash scripts will run all config files for a given experiment. Feel free to remove any of these configs if you want to speed up the process.

### Comparison

Once experiments are run, you can also plot a comparison of accuracies and losses of different experiment configurations. Compare configurations are stored in `src/compare/configs`. There you can specify whether you want to compare accuracies or losses, or both. Also, you can specify which files to compare and their corresponding configs. Use our examples to guide you! To do this navigate to the compare subfolder:

```shell
cd src/compare
```

There you can run:

```shell
python compare.py --config <config-file>.yaml
```

or

```shell
./compare.sh
```

Bare in mind that last method can run multiple compare configs at once too, as given by our example!

### Experiment Config Structure
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
  num_tasks: 3  # Number of times to repeat training with different label shufflings
  to_perturb: False

exp_desc: "mlp"  # Experiment description for output directory naming

```

In the above, we can easily customize various specs such as the model type, number of experiments, etc.
