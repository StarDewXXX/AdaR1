#!/bin/bash
# 定义模型相对路径的数组（用于--model参数）
models=(
  # your Short-CoT Model path
  # your Long-CoT Model path
  Qwen/Qwen2.5-7B-Instruct
)

model_names=(
  # Deepseek-Qwen-7B-Short-COT
  # DeepSeek-R1-Distill-Qwen-7B
  Qwen/Qwen2.5-7B-Instruct
)

# 遍历每个模型，使用数组索引获取对应的model和model_name
for i in "${!models[@]}"; do
  model="${models[i]}"
  model_name="${model_names[i]}"
  
  echo "开始评估模型: $model"
  ./scripts/eval/eval_model.sh \
    --model $model \
    --n_samples 12 \
    --response_length 16384 \
    --datasets mix_mathematic_problems \
    --output-dir model_eval/"$model_name"


  echo "模型 $model_name 的采样完成"
  echo "-----------------------------------"
done
