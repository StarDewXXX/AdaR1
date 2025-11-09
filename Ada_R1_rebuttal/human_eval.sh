#!/usr/bin/env bash
set -euo pipefail

######################### 用户可改的超参数 #########################
# MODEL_PATH="/home/user1/projects/Light-R1/deepscaler-release/models/Deepseek-Qwen-7B-NIPS/Deepseek-Qwen-7B-o1pruner-alpha-5-MATH-full-plain-v8"
# MODEL_PATH="/home/user1/projects/Light-R1/deepscaler-release/models/Deepseek-Qwen-7B-NIPS/Deepseek-Qwen-7b-dpo-v5"
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
SERVER_PORT=8000
MAX_MODEL_LEN=12288
# 这是真正存放 JSONL 的目录
RESULT_DIR="/home/user1/projects/O1-Pruner-test/Ada_R1_rebuttal/humaneval"
TEMPERATURE=1e-5        # 0→pass@1，也可改别的
N_SAMPLES=1             # 一题几份，用于 pass@k
###################################################################

# ① 根目录设为 RESULT_DIR 的父级，这样 codegen --root 后不会再多一层 dataset
ROOT_DIR="$(dirname "$RESULT_DIR")"

# ② 确保目录存在
mkdir -p "$RESULT_DIR"

# ③ 根据 EvalPlus 规则，把 MODEL_PATH 中的 “/ : # 空格” 全替换成 “--”
# SAFE_TAG=$(echo "$MODEL_PATH" | sed 's/[\/:# ]/--/g')
# 原来是这样（会把开头 "/" 也替换成 "--"）：


# 改成下面这段 —— 先去掉开头的 "/"，再替换中间的符号
REL_MODEL_PATH="${MODEL_PATH#/}"                                    # "/foo/bar" → "foo/bar"
SAFE_TAG=$(echo "$REL_MODEL_PATH" | sed 's/[\/:# ]/--/g')           # "foo/bar" → "foo--bar"

# ④ 把 1e-5 → 1e-05，与文件名匹配
TEMP_TAG=$(python3 - <<<'import sys; print(format(float(sys.argv[1]),".0e"))' "$TEMPERATURE")

# ⑤ JSONL 的完整路径（最终会落在 ROOT_DIR/humaneval/SAFE_TAG_openai_temp_TEMP_TAG.jsonl）
JSONL_FILE="${RESULT_DIR}/${SAFE_TAG}_openai_temp_${TEMP_TAG}.jsonl"

echo "[INFO] JSONL will be written to: $JSONL_FILE"

# ⑥ 确保 vLLM 不会因脚本中途退出残留
trap '[[ -n "${VLLM_PID-}" ]] && { echo "[CLEANUP] Killing vLLM ($VLLM_PID)"; kill $VLLM_PID 2>/dev/null; }' EXIT

# 1️⃣ 启动 vLLM OpenAI‑compatible server
echo "[INFO] Starting vLLM server (PID captured) …"
python -m vllm.entrypoints.openai.api_server \
       --model "$MODEL_PATH" \
       --tensor-parallel-size 4 \
       --dtype bfloat16 \
       --max-model-len "$MAX_MODEL_LEN" \
       --port "$SERVER_PORT" \
       > /home/user1/projects/O1-Pruner-test/Ada_R1_rebuttal/tmux/human_eval_vllm_server.log 2>&1 &
VLLM_PID=$!

# 2️⃣ 等待 vLLM 就绪
echo -n "[INFO] Waiting for vLLM to be ready "
until curl -s "http://localhost:${SERVER_PORT}/v1/models" >/dev/null; do
  echo -n "."
  sleep 1
done
echo " OK"

# 3️⃣ 生成代码（指定 --root，结果会写入 ROOT_DIR/humaneval）
export OPENAI_API_KEY=dummy
echo "[INFO] Running evalplus.codegen (root=$ROOT_DIR) …"
evalplus.codegen \
  --model "$MODEL_PATH" \
  --dataset humaneval \
  --backend openai \
  --base-url "http://localhost:${SERVER_PORT}/v1" \
  --temperature "$TEMPERATURE" \
  --n-samples "$N_SAMPLES" \
  --root "$ROOT_DIR"

# 4️⃣ 校验文件存在
if [[ ! -f "$JSONL_FILE" ]]; then
  echo "[ERROR] JSONL not found at $JSONL_FILE"
  exit 1
fi

# 5️⃣ 评测正确率
echo "[INFO] Running evalplus.evaluate …"
evalplus.evaluate \
  --dataset humaneval \
  --samples "$JSONL_FILE"

# 6️⃣ 统计平均 completion tokens
echo "[INFO] Computing average completion tokens …"
python3 - <<PY
import json, statistics, pathlib
from transformers import AutoTokenizer

jsonl = pathlib.Path("$JSONL_FILE")
tok   = AutoTokenizer.from_pretrained("$MODEL_PATH", use_fast=True)

lengths = []
for line in jsonl.open():
    obj = json.loads(line)
    ct = (obj.get("usage") or {}).get("completion_tokens")
    if ct is None:
        ct = len(tok(obj["completion"]).input_ids)
    lengths.append(ct)

print("="*30)
print(f"Average completion tokens: {statistics.mean(lengths):.2f}")
print("="*30)
PY

# 7️⃣ 清理 vLLM
echo "[INFO] Shutting down vLLM (PID $VLLM_PID) …"
kill "$VLLM_PID"
wait "$VLLM_PID" 2>/dev/null || true
echo "[INFO] All done."
