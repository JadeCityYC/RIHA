#!/bin/bash
# Train RIHA on the IU X-Ray dataset.
python main.py \
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
