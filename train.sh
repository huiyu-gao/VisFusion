#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

python main.py --cfg ./config/train.yaml TRAIN.EPOCHS 20 MODEL.FUSION.FUSION_ON False
python main.py --cfg ./config/train.yaml TRAIN.EPOCHS 41 TRAIN.FINETUNE_LAYER None MODEL.PASS_LAYERS 2
python main.py --cfg ./config/train.yaml TRAIN.EPOCHS 44 TRAIN.FINETUNE_LAYER 0 MODEL.PASS_LAYERS 0
python main.py --cfg ./config/train.yaml TRAIN.EPOCHS 47 TRAIN.FINETUNE_LAYER 1 MODEL.PASS_LAYERS 1
python main.py --cfg ./config/train.yaml TRAIN.EPOCHS 50 TRAIN.FINETUNE_LAYER 2 MODEL.PASS_LAYERS 2
