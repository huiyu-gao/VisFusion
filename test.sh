#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

python main.py --cfg ./config/test.yaml

python tools/evaluation.py --model ./results/scene_scannet_checkpoints_fusion_eval_49
