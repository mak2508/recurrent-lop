#!/bin/bash

python ./lang-reshuffle.py --config ./configs/gru_small.yaml
python ./lang-reshuffle.py --config ./configs/gru_medium.yaml
python ./lang-reshuffle.py --config ./configs/gru_large.yaml
python ./lang-reshuffle.py --config ./configs/gru_cbp.yaml
python ./lang-reshuffle.py --config ./configs/gru_l2.yaml
python ./lang-reshuffle.py --config ./configs/gru_perturb.yaml

