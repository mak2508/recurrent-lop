#!/bin/bash
# every job takes approx 3 hours

python ./mnist-reshuffle.py --config ./configs/mlp.yaml
python ./mnist-reshuffle.py --config ./configs/mlp_l2.yaml
python ./mnist-reshuffle.py --config ./configs/mlp_cbp.yaml
python ./mnist-reshuffle.py --config ./configs/mlp_perturb.yaml
python ./mnist-reshuffle.py --config ./configs/lstm_perturb.yaml
python ./mnist-reshuffle.py --config ./configs/lstm_perturb.yaml