#!/bin/sh
python train_dance.py --config $2 --source ./txt/source_amazon_cls_40AN.txt --target ./txt/target_dslr_cls.txt --gpu $1
python train_dance.py --config $2 --source ./txt/source_amazon_cls_40AN.txt --target ./txt/target_webcam_cls.txt --gpu $1
python train_dance.py --config $2 --source ./txt/source_dslr_cls_40AN.txt --target ./txt/target_webcam_cls.txt --gpu $1
python train_dance.py --config $2 --source ./txt/source_dslr_cls_40AN.txt --target ./txt/target_amazon_cls.txt --gpu $1
python train_dance.py --config $2 --source ./txt/source_webcam_cls_40AN.txt --target ./txt/target_amazon_cls.txt --gpu $1
python train_dance.py --config $2 --source ./txt/source_webcam_cls_40AN.txt --target ./txt/target_dslr_cls.txt --gpu $1
