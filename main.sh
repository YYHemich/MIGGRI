#!/usr/bin/env bash
python train.py --bidirectional
python retrain.py --model model/mean_sage_dot_testIdx3_best.pkl
