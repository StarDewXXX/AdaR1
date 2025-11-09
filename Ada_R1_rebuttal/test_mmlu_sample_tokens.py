# -*- coding: utf-8 -*-
import argparse, json, os, time, re, random
import numpy as np, pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from categories import categories, subcategories

CHOICES = ["A", "B", "C", "D"]

def extract_boxed_answer(text: str) -> str:
    m = re.findall(r'\\boxed{([A-D])}', text, flags=re.IGNORECASE)
    return m[-1].upper() if m else ""

def format_example(df: pd.DataFrame, idx: int, include_answer: bool = True) -> str:
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += f"\n{CHOICES[j]}. {df.iloc[idx, j + 1]}"
    prompt += "\nAnswer:"
    if include_answer:
        prompt += f" {df.iloc[idx, k + 1]}\n\n"
    return prompt

def gen_prompt(train_df: pd.DataFrame, subject: str, k: int = -1) -> str:
    header = "Let's think step by step and output the final answer within \\boxed{}. (e.g., \\boxed{A})\n\n"
    if k == -1:
        k = train_df.shape[0]
    body = "".join(format_example(train_df, i) for i in range(k))
    return header + body

def build_prompts_for_subject(
    tokenizer,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ntrain: int,
    max_ctx_len: int,
):
    """为该 subject 的所有题目一次性构造 prompts，并做长度裁剪。"""
    prompts = []
    k_used_list = []
    for i in tqdm(range(test_df.shape[0]), desc="Building prompts"):
        k = ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject="unused", k=k)
        prompt = train_prompt + prompt_end

        # 按 tokenizer 长度裁剪
        while len(tokenizer.encode(prompt, add_special_tokens=False)) > max_ctx_len and k > 0:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject="unused", k=k)
            prompt = train_prompt + prompt_end

        prompts.append(prompt)
        k_used_list.append(k)
    return prompts, k_used_list

def eval_one_subject_batched(
    args,
    subject: str,
    llm: LLM,
    tokenizer,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
):  
    print('=' * 20, f"Evaluating subject: {subject}", '=' * 20)

    # 构造所有 prompts
    prompts, k_used_list = build_prompts_for_subject(
        tokenizer, dev_df, test_df, args.ntrain, args.max_ctx_len
    )

    # 统一采样参数
    params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature if args.do_sample else 0.0,
        top_p=args.top_p if args.do_sample else 1.0,
        top_k=args.top_k if (args.do_sample and args.top_k > 0) else -1,
        logprobs=0,
        prompt_logprobs=0,
        detokenize=True,
    )

    # **一次性生成**
    outputs = llm.generate(prompts, params, use_tqdm=True)

    answers = CHOICES[: test_df.shape[1] - 2]
    cors, gen_lens = [], []

    # vLLM 会按输入顺序返回
    in_valid = []
    for i, out in enumerate(outputs):
        if len(out.outputs) == 0:
            text, token_ids = "", []
        else:
            seq = out.outputs[0]
            text, token_ids = seq.text, seq.token_ids

        gen_lens.append(len(token_ids))
        pred = extract_boxed_answer(text.strip())
        label = test_df.iloc[i, test_df.shape[1] - 1]

        if pred not in answers:
            false_options = [c for c in answers if c != label]
            # pred = "A"
            pred = random.choice(false_options)
            in_valid.append(1)

        # label = test_df.iloc[i, test_df.shape[1] - 1]
        cors.append(pred == label)

    cors = np.array(cors)
    acc = cors.mean()
    print(f"Average accuracy {acc:.3f} - {subject}")
    return cors, np.array(gen_lens), k_used_list, in_valid

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem_util,
        trust_remote_code=True,
        dtype="half",
        # 可选：max_model_len=args.max_ctx_len
    )

    # subjects = sorted(
    #     f.split("_test.csv")[0]
    #     for f in os.listdir(os.path.join(args.data_dir, "test"))
    #     if f.endswith("_test.csv")
    # )
    all_files = os.listdir(os.path.join(args.data_dir, "test"))
    subjects = sorted(
    subject_name
    for f in all_files if f.endswith("_test.csv")
    for subject_name in [f.split("_test.csv")[0]]
    if subject_name in subcategories  # 只保留在 subcategories 里的
    )


    os.makedirs(args.save_dir, exist_ok=True)
    result_dir = os.path.join(args.save_dir, f"results_{os.path.basename(args.model)}")
    os.makedirs(result_dir, exist_ok=True)

    all_cors = []
    cat_cors = {cat: [] for cat in categories}
    cat_genlens = {cat: [] for cat in categories}
    cat_invalid = {cat: [] for cat in categories}

    t0 = time.time()
    for subject in subjects:
        dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", f"{subject}_dev.csv"), header=None)[: args.ntrain]
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", f"{subject}_test.csv"), header=None)

        cors, gen_lens, k_used, in_valid = eval_one_subject_batched(
            args, subject, llm, tokenizer, dev_df, test_df
        )
        all_cors.append(cors)

        # 分类汇总
        for subcat in subcategories[subject]:
            for cat_name, subcat_list in categories.items():
                if subcat in subcat_list:
                    cat_cors[cat_name].append(cors)
                    cat_genlens[cat_name].append(gen_lens)
                    cat_invalid[cat_name].append(in_valid)

        # 保存 CSV（增加 k_used，便于查看裁剪情况）
        test_df[f"{os.path.basename(args.model)}_correct"] = cors
        test_df[f"{os.path.basename(args.model)}_gen_len"] = gen_lens
        test_df[f"{os.path.basename(args.model)}_k_used"] = k_used
        test_df.to_csv(os.path.join(result_dir, f"{subject}.csv"), index=False)

    # 汇总
    results = {"categories": {}, "accuracy": {}, "avg_gen_tokens": {}}
    for cat in categories:
        cat_acc = np.concatenate(cat_cors[cat]).mean()
        cat_inv = np.concatenate(cat_invalid[cat]).sum() # / len(cat_invalid[cat])
        cat_tok = np.concatenate(cat_genlens[cat]).mean()
        results["accuracy"][cat] = float(cat_acc)
        results["avg_gen_tokens"][cat] = float(cat_tok)
        print(f"[{cat}]  acc={cat_acc:.3f}  avg_gen_tokens={cat_tok:.2f}  invalid={cat_inv:.3f}")
    print(f"MODEL NAME: {args.model}")
    weighted_acc = np.concatenate(all_cors).mean()
    weighted_tok = np.mean(np.concatenate([np.array(g) for g in cat_genlens.values()]))
    results["weighted_accuracy"] = float(weighted_acc)
    results["weighted_generation_tokens"] = float(weighted_tok)
    results["cost_time_sec"] = float(time.time() - t0)

    json_path = os.path.join(args.save_dir, f"metrics_{args.model.replace('/', '_')}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n=== OVERALL ===\naccuracy={weighted_acc:.3f} | avg_gen_tokens={weighted_tok:.2f}")
    print(f"Saved to {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=0)
    parser.add_argument("--data_dir", "-d", type=str, default="/home/user1/.cache/huggingface/hub/MMLU/data")
    parser.add_argument("--save_dir", "-s", type=str, default="/home/user1/projects/O1-Pruner-test/Ada_R1_rebuttal/mmlu_output/with_sample")
    parser.add_argument("--model", "-m", type=str, default='/home/user1/projects/Light-R1/deepscaler-release/models/Deepseek-Qwen-7B/Deepseek-Qwen-7B-merge-0.8-o1pruner')
    # parser.add_argument("--model", "-m", type=str, default='/home/user1/projects/Light-R1/deepscaler-release/models/Deepseek-Qwen-7B/Deepseek-Qwen-7B-dpo')
    # parser.add_argument("--model", "-m", type=str, default='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')

    # vLLM
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--gpu_mem_util", type=float, default=0.90)
    parser.add_argument("--max_ctx_len", type=int, default=4096)

    # 生成
    parser.add_argument("--do_sample", default=False, action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=8192)

    args = parser.parse_args()
    main(args)
