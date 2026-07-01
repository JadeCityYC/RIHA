#!/bin/bash
# Evaluate a trained RIHA checkpoint on IU X-Ray.
# Set --resume to the checkpoint you want to evaluate.
python test.py \
    --image_dir data/iu_xray/images/ \
    --ann_path data/iu_xray/annotation.json \
    --dataset_name iu_xray \
    --max_seq_length 60 \
    --threshold 3 \
    --batch_size 16 \
    --epochs 100 \
    --save_dir results/iu_xray \
    --step_size 50 \
    --gamma 0.1 \
    --seed 9223 \
    --resume results/iu_xray/model_best.pth
