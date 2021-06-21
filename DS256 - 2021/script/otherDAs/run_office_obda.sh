#!/bin/sh
#python train_dance.py --config $2 --source ./txt/source_amazon_obda_45AN.txt --target ./txt/target_dslr_obda.txt --gpu $1
python train_dance.py --config $2 --source ./txt/source_amazon_obda_45AN.txt --target ./txt/target_webcam_obda.txt --gpu $1
#python train_dance.py --config $2 --source ./txt/source_dslr_obda_45AN.txt --target ./txt/target_webcam_obda.txt --gpu $1
python train_dance.py --config $2 --source ./txt/source_dslr_obda_45SN.txt --target ./txt/target_amazon_obda.txt --gpu $1
#python train_dance.py --config $2 --source ./txt/source_webcam_obda_45AN.txt --target ./txt/target_amazon_obda.txt --gpu $1
python train_dance.py --config $2 --source ./txt/source_webcam_obda_45AN.txt --target ./txt/target_dslr_obda.txt --gpu $1
