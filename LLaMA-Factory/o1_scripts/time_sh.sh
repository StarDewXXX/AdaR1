#!/bin/bash

# 定义多组 model_name 和 model_path
model_names=("Marco-dpo-iter0-beta-0.1" "Marco-loft-iter0-alpha_2-ppo-normed" "Marco-sft-shortest-iter0" "Marco-o1" "Marco-o1-fast-prompt")
model_paths=("saves/math/Marco-dpo-iter0-beta-0.1" "saves/math/Marco-loft-iter0-alpha_2-ppo-normed" "saves/math/Marco-sft-shortest-iter0" "AIDC-AI/Marco-o1" "AIDC-AI/Marco-o1")
modes=("normal" "normal" "normal" "normal" "very_fast")


# 检查数组长度是否一致
if [ ${#model_names[@]} -ne ${#model_paths[@]} ]; then
  echo "Error: model_names and model_paths must have the same length"
  exit 1
fi

# 循环处理每一组模型
for i in "${!model_names[@]}"; do
  model_name=${model_names[$i]}
  model_path=${model_paths[$i]}
  mode=${modes[$i]}
  
  echo "Running inference for model: $model_name"
  CUDA_VISIBLE_DEVICES=7 HF_ENDPOINT=https://hf-mirror.com python v4_scripts/inference.py \
    --K 5 \
    --num_samples 500 \
    --dataset math_test \
    --model ${model_name} \
    --speed normal \
    --model_path ${model_path} \
    --save_output False \
    --speed ${mode} \
    --n_gpus 1 \
    --shuffle True
done