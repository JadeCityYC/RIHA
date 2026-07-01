#!/bin/bash
# Train RIHA on the MIMIC-CXR dataset.
python main.py \
    --image_dir data/mimic_cxr/images \
    --ann_path data/mimic_cxr/mimic-annotation.json \
    --dataset_name mimic_cxr \
    --max_seq_length 100 \
    --threshold 10 \
    --batch_size 64 \
    --epochs 30 \
    --save_dir results/mimic_cxr \
    --step_size 1 \
    --gamma 0.8 \
    --seed 456789 \
    --use_alignment_loss \
    --alignment_weight 0.5 \
    --high_level_weight 0.5 \
    --mid_level_weight 0.2 \
    --low_level_weight 0.3 \
    --ot_impl pot-uot-l2 \
    --ot_reg 0.1 \
    --ot_tau 0.5 \
    --paragraph_features_path data/features/paragraph_features.json \
    --sentence_features_path data/features/sentence_features.json \
    --word_features_path data/features/word_features.json
