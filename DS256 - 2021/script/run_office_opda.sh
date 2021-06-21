#!/bin/sh
python train_dance_co.py --config $2 --source ./txt/source_amazon_opda_20SN.txt --target ./txt/target_dslr_opda.txt --gpu $1 --relationship_path relationship/source_amazon_opda_20SN.npy
python train_dance_co.py --config $2 --source ./txt/source_amazon_opda_20SN.txt --target ./txt/target_webcam_opda.txt --gpu $1 --relationship_path relationship/source_amazon_opda_20SN.npy
python train_dance_co.py --config $2 --source ./txt/source_dslr_opda_20SN.txt --target ./txt/target_webcam_opda.txt --gpu $1 --relationship_path relationship/source_dslr_opda_20SN.npy
python train_dance_co.py --config $2 --source ./txt/source_dslr_opda_20SN.txt --target ./txt/target_amazon_opda.txt --gpu $1 --relationship_path relationship/source_dslr_opda_20SN.npy
python train_dance_co.py --config $2 --source ./txt/source_webcam_opda_20SN.txt --target ./txt/target_amazon_opda.txt --gpu $1 --relationship_path relationship/source_webcam_opda_20SN.npy
python train_dance_co.py --config $2 --source ./txt/source_webcam_opda_20SN.txt --target ./txt/target_dslr_opda.txt --gpu $1 --relationship_path relationship/source_webcam_opda_20SN.npy
