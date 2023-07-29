#!/bin/bash
# -*- coding: utf-8 -*-
#sub -q 83a100ib -gpu "num=3:aff=yes" -o output1.log -e error1.log < ./scripts/guji_pretrain.sh
module use /fsa/home/hqz_zhangjd/rebuild
module load rebuild1
# 设置OMP_NUM_THREADS环境变量
export OMP_NUM_THREADS=2
# 多卡
export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch --config_file=./configs/default_config.yaml src/train_pt.py \
    --model_name_or_path ../../models/Ziya-LLaMA-13B-v1/ \
    --prompt_template ziya \
    --do_train \
    --dataset guji_pretrain \
    --dataset_dir ./data \
    --finetuning_type lora \
    --lora_rank 16 \
    --output_dir checkpoints/guji/pretrain_7-13 \
    --overwrite_cache \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 1000 \
    --save_steps 2000 \
    --learning_rate 6e-5 \
    --num_train_epochs 5.0 \
    --plot_loss \
    --fp16