#!/usr/bin/env bash
set -u  # 避免未定义变量；不使用 -e，或在每条命令后用 || true 继续

LOG_DIR="/home/user1/projects/O1-Pruner-test/Ada_R1_rebuttal/tmux"
mkdir -p "$LOG_DIR"

echo "$(date)  等待 8 小时后开始运行……"
sleep 8h

python /home/user1/projects/O1-Pruner-test/Ada_R1_rebuttal/test_mmlu_sample_tokens.py --model /home/user1/projects/Light-R1/deepscaler-release/models/Deepseek-Qwen-7B/Deepseek-Qwen-7B-merge-0.8-dpo-beta-0.1-no-ln-bilevel-fulldata-M1-4-M2-2-cosine-lr \
  >> "$LOG_DIR/Deepseek-Qwen-7B-merge-0.8-dpo-beta-0.1-no-ln-bilevel-fulldata-M1-4-M2-2-cosine-lr.log" 2>&1 || true

python /home/user1/projects/O1-Pruner-test/Ada_R1_rebuttal/test_mmlu_sample_tokens.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  >> "$LOG_DIR/DeepSeek-R1-Distill-Qwen-7B.log" 2>&1 || true

python /home/user1/projects/O1-Pruner-test/Ada_R1_rebuttal/test_mmlu_sample_tokens.py --model /home/user1/projects/Light-R1/deepscaler-release/models/Deepseek-Qwen-7B-NIPS/Deepseek-Qwen-7B-o1pruner-alpha-5-MATH-full-plain-v8 \
  >> "$LOG_DIR/Deepseek-Qwen-7B-o1pruner-alpha-5-MATH-full-plain-v8.log" 2>&1 || true

python /home/user1/projects/O1-Pruner-test/Ada_R1_rebuttal/test_mmlu_sample_tokens.py --model /home/user1/projects/Light-R1/deepscaler-release/models/Deepseek-Qwen-7B-NIPS/Deepseek-Qwen-7b-dpo-v5 \
  >> "$LOG_DIR/Deepseek-Qwen-7b-dpo-v5.log" 2>&1 || true

echo "$(date)  全部任务执行完毕。"
