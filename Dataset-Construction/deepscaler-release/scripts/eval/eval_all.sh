#!/bin/bash
# 定义模型相对路径的数组（用于--model参数）
models=(
  # local path
  Qwen/Qwen2.5-7B-Instruct
)

model_names=(
  Qwen/Qwen2.5-7B-Instruct
)

# 遍历每个模型，使用数组索引获取对应的model和model_name
for i in "${!models[@]}"; do
  model="${models[i]}"
  model_name="${model_names[i]}"
  
  echo "开始评估模型: $model"

  # ./scripts/eval/eval_model.sh \
  #   --model $model \
  #   --n_samples 12 \
  #   --response_length 16384 \
  #   --datasets math_train \
  #   --output-dir model_eval/"$model_name"

  # ./scripts/eval/eval_model.sh \
  #   --model $model \
  #   --n_samples 12 \
  #   --response_length 16384 \
  #   --datasets mix_mathematic_problems \
  #   --output-dir model_eval/"$model_name"

  ./scripts/eval/eval_model.sh \
    --model $model \
    --n_samples 1 \
    --response_length 12288 \
    --datasets math \
    --output-dir model_eval/"$model_name"
  
  # 针对 aime25 数据集的评估
  # ./scripts/eval/eval_model.sh \
  #   --model $model \
  #   --n_samples 4 \
  #   --response_length 16384 \
  #   --datasets aime25 \
  #   --output-dir model_eval/"$model_name"
  
  # 针对 gsm8k 数据集的评估
  # ./scripts/eval/eval_model.sh \
  #   --model $model \
  #   --n_samples 1 \
  #   --response_length 4096 \
  #   --datasets gsm8k \
  #   --output-dir model_eval/"$model_name"

    ./scripts/eval/eval_model.sh \
    --model $model \
    --n_samples 1 \
    --response_length 12288 \
    --datasets minerva \
    --output-dir model_eval/"$model_name"

    ./scripts/eval/eval_model.sh \
    --model $model \
    --n_samples 1 \
    --response_length 12288 \
    --datasets olympiad_bench \
    --output-dir model_eval/"$model_name"

    #   ./scripts/eval/eval_model.sh \
    # --model $model \
    # --n_samples 1 \
    # --response_length 16384 \
    # --datasets gpqa \
    # --output-dir model_eval/"$model_name"

  echo "模型 $model_name 的评估完成"
  echo "-----------------------------------"
done
