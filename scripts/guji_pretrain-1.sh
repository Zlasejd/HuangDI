#!/bin/bash
#sub -q 83a100ib -gpu "num=3:aff=yes" -o output.log -e error.log < ./scripts/guji_pretrain.sh
module load cuda/12.0.0
module use /fsa/home/hqz_zhangjd/rebuild
module load rebuild1
# 设置OMP_NUM_THREADS环境变量

python src/train_pt.py \
    --model_name_or_path ../../models/Ziya-LLaMA-13B-v1/ \
    --prompt_template ziya \
    --do_train \
    --dataset guji_pretrain \
    --finetuning_type lora \
    --lora_rank 16 \
    --output_dir checkpoints/guji/pretrain_7-07-1 \
    --overwrite_cache \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 200 \
    --save_steps 2000 \
    --learning_rate 1e-4 \
    --num_train_epochs 4.0 \
    --plot_loss \
    --fp16
