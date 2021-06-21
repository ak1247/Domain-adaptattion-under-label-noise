#!/bin/sh
python train_dance.py --config $2 --source ./txt/source_amazon_opda_45AN.txt --target ./txt/target_dslr_opda.txt --gpu $1
python train_dance.py --config $2 --source ./txt/source_amazon_opda_45AN.txt --target ./txt/target_webcam_opda.txt --gpu $1
python train_dance.py --config $2 --source ./txt/source_dslr_opda_45AN.txt --target ./txt/target_webcam_opda.txt --gpu $1
python train_dance.py --config $2 --source ./txt/source_dslr_opda_45AN.txt --target ./txt/target_amazon_opda.txt --gpu $1
python train_dance.py --config $2 --source ./txt/source_webcam_opda_45AN.txt --target ./txt/target_amazon_opda.txt --gpu $1
python train_dance.py --config $2 --source ./txt/source_webcam_opda_45AN.txt --target ./txt/target_dslr_opda.txt --gpu $1
