#!/bin/bash

python ./mnist-binary.py --config ./configs/mlp.yaml
python ./mnist-binary.py --config ./configs/mlp_perturb.yaml
python ./mnist-binary.py --config ./configs/lstm.yaml
python ./mnist-binary.py --config ./configs/lstm_perturb.yaml
