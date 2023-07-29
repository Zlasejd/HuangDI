#!/bin/bash

#CUDA_VISIBLE_DEVICES=0 python ../src/cli_demo.py \
#    --model_name_or_path /home2/mrwu/models/chinese-llama-plus-7b

#CUDA_VISIBLE_DEVICES=0 python ./src/cli_demo.py \
#    --model_name_or_path /root/autodl-tmp/Ziya-LLaMA-13B-v1 \
#    --prompt_template ziya \
#    --repetition_penalty 1.2


CUDA_VISIBLE_DEVICES=0 python ./src/cli_demo.py \
    --model_name_or_path /root/autodl-tmp/Ziya-LLaMA-13B-v1 \
    --checkpoint_dir ./checkpoints/medical_6-19 \
    --prompt_template ziya \
    --repetition_penalty 1.2