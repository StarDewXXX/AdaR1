import json, statistics, pathlib
from transformers import AutoTokenizer
from tqdm import tqdm
import os
# hf mirror
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# MODEL_PATH="/home/user1/projects/Light-R1/deepscaler-release/models/Deepseek-Qwen-7B-NIPS/Deepseek-Qwen-7B-o1pruner-alpha-5-MATH-full-plain-v8"
# JSONL_FILE = "/home/user1/projects/O1-Pruner-test/o1_scripts/humaneval/o1_pruner_humaneval_12288_normal_K-1.jsonl"
MODEL_PATH="/home/user1/projects/Light-R1/deepscaler-release/models/Deepseek-Qwen-7B-NIPS/Deepseek-Qwen-7b-dpo-v5"
JSONL_FILE = "/home/user1/projects/O1-Pruner-test/o1_scripts/humaneval/dpo_humaneval_12288_normal_K-1.jsonl"
# MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# JSONL_FILE="/home/user1/projects/O1-Pruner-test/o1_scripts/humaneval/DeepSeek-R1-Distill-Qwen-7B_humaneval_12288_normal_K-1.jsonl"

jsonl = pathlib.Path(JSONL_FILE)
tok   = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)

lengths = []
# for line in jsonl.open():
for i, line in enumerate(tqdm(jsonl.open())):
    obj = json.loads(line)
    ct = (obj.get("usage") or {}).get("completion_tokens")
    # print(ct)
    if ct is None:
        ct = len(tok(obj["solution"]).input_ids)
    lengths.append(ct)
print(f"Total samples: {len(lengths)}")
print(f"Model: {MODEL_PATH}")
print("="*30)
print(f"Average completion tokens: {statistics.mean(lengths):.2f}")
print("="*30)