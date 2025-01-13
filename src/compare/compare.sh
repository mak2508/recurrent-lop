#!/bin/bash

python compare.py --config ./configs/compare_mnist_binary_baseline.yaml

python compare.py --config ./configs/compare_mnist_reshuffle_mlp_l2.yaml
python compare.py --config ./configs/compare_mnist_reshuffle_mlp.yaml
python compare.py --config ./configs/compare_mnist_reshuffle_lstm.yaml
python compare.py --config ./configs/compare_mnist_reshuffle_baseline.yaml

python compare.py --config ./configs/compare_lang_reshuffle_gru_sizes.yaml
python compare.py --config ./configs/compare_lang_reshuffle_gru_mitigations.yaml

python compare.py --config ./configs/compare_best_gru_mlp.yaml

