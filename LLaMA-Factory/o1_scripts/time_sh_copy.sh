#!/bin/bash

# 定义多组 model_name 和 model_path
# model_names=("QwQ" "QwQ-fast-prompt" "QwQ-loft" "QwQ-dpo")
# model_paths=("Qwen/QwQ-32B-Preview" "Qwen/QwQ-32B-Preview" "saves/math/QwQ-loft-iter0-alpha_5" "saves/math/QwQ-dpo")
# modes=("normal" "very_fast" "normal" "normal")
model_names=("QwQ-loft" "QwQ-dpo")
model_paths=("saves/math/QwQ-loft-iter0-alpha_5" "saves/math/QwQ-dpo")
modes=("normal" "normal")
# model_names=("QwQ-sft")
# model_paths=("saves/math/QwQ-sft")
# modes=("normal")

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
  CUDA_VISIBLE_DEVICES=4,5,6,7 HF_ENDPOINT=https://hf-mirror.com python v4_scripts/inference.py \
    --K 3 \
    --num_samples 500 \
    --dataset math_test \
    --model ${model_name} \
    --speed normal \
    --model_path ${model_path} \
    --save_output False \
    --speed ${mode} \
    --n_gpus 4 \
    --shuffle True
done