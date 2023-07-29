#!/bin/bash
# -*- coding: utf-8 -*-
#sub -q 83a100ib -gpu "num=4:aff=yes" -o output1.log -e error1.log < ./scripts/guji_pretrain.sh
module use /fsa/home/hqz_zhangjd/rebuild
module load rebuild1
# 设置OMP_NUM_THREADS环境变量
#export OMP_NUM_THREADS=2
# 设置OMP_NUM_THREADS环境变量
#export CUDA_VISIBLE_DEVICES=0,1,2,3
#accelerate launch --config_file=./configs/infer_config.yaml src/train_sft.py \
CUDA_VISIBLE_DEVICES=0 python src/train_sft.py \
    --model_name_or_path ../../models/Ziya-LLaMA-13B-v1/ \
    --prompt_template ziya \
    --do_eval \
    --dataset guji-sft \
    --dataset_dir ./data \
    --finetuning_type lora \
    --lora_rank 16 \
    --checkpoint_dir ./checkpoints/guji/sft_7-13 \
    --output_dir ./checkpoints/guji/evaluation_养生食疗类 \
    --per_device_eval_batch_size 4 \
    --max_steps 100 \
    --predict_with_generate

