#!/bin/bash
# -*- coding: utf-8 -*-
#sub -q 83a100ib -gpu "num=4:aff=yes" -o output1.log -e error1.log < ./scripts/guji_pretrain.sh
module use /fsa/home/hqz_zhangjd/rebuild
module load rebuild1
# 设置OMP_NUM_THREADS环境变量
export OMP_NUM_THREADS=2
# 多卡
export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch --config_file=./configs/default_config.yaml src/train_sft.py \
    --do_train \
    --model_name_or_path ../../models/Ziya-LLaMA-13B-v1/ \
    --checkpoint_dir ./checkpoints/guji/pretrain_7-7 \
    --resume_lora_training False \
    --dataset guji-sft \
    --dataset_dir ./data \
    --finetuning_type lora \
    --lora_rank 16 \
    --output_dir ./checkpoints/guji/sft_7-13 \
    --overwrite_cache \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --evaluation_strategy steps \
    --logging_steps 100 \
    --save_steps 5000 \
    --dev_ratio 0.05 \
    --learning_rate 3e-4 \
    --resume_lora_training False \
    --num_train_epochs 6.0 \
    --load_best_model_at_end \
    --fp16 \
    --plot_loss True \
    --ddp_find_unused_parameters False