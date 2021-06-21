#!/bin/sh
python train_dance_co.py --config configs/visda-train-config_UDA.yaml --source ./txt/source_visda_univ_45SN.txt --target ./txt/target_visda_univ.txt --gpu $1 --relationship_path relationship/source_visda_univ_45SN.npy
