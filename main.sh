#!/usr/bin/env bash
python train.py --bidirectional
python retrain.py --model model/lstm_sage_dot_testIdx2_best.pkl
