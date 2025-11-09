import re, sys, os, statistics, random, json
from datasets import load_dataset
from vllm import LLM, SamplingParams

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

ds = load_dataset("lucasmccabe/logiqa", split="test", trust_remote_code=True)
LETTERS = "ABCD"
# MODEL_PATH = '/home/user1/projects/Light-R1/deepscaler-release/models/Deepseek-Qwen-7B/Deepseek-Qwen-7B-merge-0.8-dpo-beta-0.1-no-ln-bilevel-fulldata-M1-4-M2-2-cosine-lr'
# MODEL_PATH = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
# MODEL_PATH = '/home/user1/projects/Light-R1/deepscaler-release/models/Deepseek-Qwen-7B/Deepseek-Qwen-7B-merge-0.8-o1pruner'
MODEL_PATH = '/home/user1/projects/Light-R1/deepscaler-release/models/Deepseek-Qwen-7B/Deepseek-Qwen-7B-dpo'
def build_prompt(ex):
    opts = "\n".join(f"{LETTERS[i]}. {ex['options'][i]}" for i in range(4))
    return (
        # "You are solving a logical reasoning multiple-choice problem.\n"
        # "Let's think step by step and output the final answer within \\boxed{}.\n"
        # "Only one of A, B, C, or D is correct. Example: \\boxed{A}\n\n"
        f"Context:\n{ex['context']}\n\n"
        f"Question:\n{ex['query']}\n\n"
        f"Options:\n{opts}\n\n"
        "Let's think step by step and output the final answer within \\boxed{}. (e.g., \\boxed{A})\n\n"
    )

prompts = [build_prompt(ex) for ex in ds]

print(f"Total {len(prompts)} prompts built.")
print(f"Example prompt:\n{prompts[0]}")

# for ex in ds:
#     print("=" * 20)
#     print(ex)
#     sys.exit()
# sys.exit()

llm = LLM(model=MODEL_PATH,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.9,
        )
temperature = 0.7
top_p = 0.9
top_k = -1
do_sample = False
sampling = SamplingParams(
    temperature=temperature if do_sample else 0.0,
    top_p=top_p if do_sample else 1.0,
    top_k=top_k if (do_sample and top_k > 0) else -1,
    max_tokens=8192,   # 建议先用较保守的上限
)

# 统计提示 tokens（可选）
tokenizer = llm.get_tokenizer()
prompt_token_lens = [len(tokenizer.encode(p)) for p in prompts]

outputs = llm.generate(prompts, sampling, use_tqdm=True)

# def pick_letter(text: str):
#     m = re.search(r"\\boxed\{\s*([A-D])\s*\}", text, flags=re.IGNORECASE)
#     if m: return m.group(1).upper()
#     m = re.search(r"\b([A-D])\b", text.strip(), flags=re.IGNORECASE)
#     return m.group(1).upper() if m else None

def pick_letter(text: str) -> str:
    m = re.findall(r'\\boxed{([A-D])}', text, flags=re.IGNORECASE)
    return m[-1].upper() if m else None
error_data = []
save_error_data_path = f'{MODEL_PATH}_error_data.json'
pred, gold, invalid = [], [], 0
gen_token_lens = []
# invalid = 0
for out, ex in zip(outputs, ds):
    text = out.outputs[0].text
    gen_ids = out.outputs[0].token_ids    # 生成段 token 序列
    gen_token_lens.append(len(gen_ids))

    letter = pick_letter(text)
    if letter is None:
        true_answer = ex["correct_option"]
        # 随机生成ABCD中不等于true_answer的一个字母
        choices = ["A", "B", "C", "D"]
        false_options = [c for c in choices if c != true_answer]
        letter = random.choice(false_options)  # 随机挑一个不等于 true_answer 的字母
        print('===========example:===========\n', ex)
        print('===========text:===========\n', text[-100:]) 
        invalid += 1
        # letter = "A"
        print("Invalid output")
        error_data.append({
            "context": ex["context"],
            "query": ex["query"],
            "options": ex["options"],
            "llm_output": text[-100:],
            "true_answer": true_answer,
        })
        
    pred.append(LETTERS.index(letter))
    gold.append(ex["correct_option"])

acc = sum(int(p==g) for p,g in zip(pred,gold)) / len(ds)

avg_gen = statistics.mean(gen_token_lens)
p50_gen = statistics.median(gen_token_lens)

avg_prompt = statistics.mean(prompt_token_lens)
avg_total = statistics.mean([a+b for a,b in zip(prompt_token_lens, gen_token_lens)])

print(f"Test size={len(ds)}  Acc={acc:.4f}  Invalid={invalid}")
print(f"Gen tokens: avg={avg_gen:.1f}, 中位数={p50_gen}, max={max(gen_token_lens)}")
print(f"Prompt tokens: avg={avg_prompt:.1f}")
print(f"Total tokens (prompt+gen): avg={avg_total:.1f}")
print(f"Model: {MODEL_PATH}")

with open(save_error_data_path, "w", encoding="utf-8") as f:
    for record in error_data:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Error data saved to {save_error_data_path}")
