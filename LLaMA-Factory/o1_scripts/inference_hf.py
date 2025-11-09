import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import random
from utils import extract_answer, get_example
from grader import grade_answer
from collections import defaultdict
from datasets import load_dataset
import copy
import argparse
import json
import time
import os
import sys
import re
     


def find_all_equivalence_classes(objects, equivalence_func):
    # 字典用来记录等价类代表元素和等价类的所有元素
    equivalence_classes = {}

    for obj in objects:
        found_class = False
        for rep in equivalence_classes.keys():
            # 判断当前对象是否和已有代表元素等价
            if equivalence_func(obj, rep):
                equivalence_classes[rep].append(obj)
                found_class = True
                break
        # 如果没有找到等价的代表元素，则将当前对象作为新的等价类的代表
        if not found_class:
            equivalence_classes[obj] = [obj]

    # 按等价类大小排序输出
    sorted_classes = sorted(equivalence_classes.values(), key=len, reverse=True)
    return sorted_classes


def prepare_prompts_for_solution(question, model, speed=None):
    
    if speed == None:
        speed_prompt = ""
    
    if speed == "very_fast":
        speed_prompt = "\nThis is an easy problem. Please solve it qucikly without any pause, check or reflection."
    if speed == "fast":
        speed_prompt = "\nBe confident. Solving this problem quickly with less pause, stop or reflection."
    if speed == "normal":
        speed_prompt = ""
    if speed == "slow":
        speed_prompt = "\nThis problem is hard. So you need to think rigorously and do more verifications and checks until you are absolutely confident about your answer."

    if model == "Marco" or "Marco" in model:
        prompt = [
                {"role": "system", "content": f"You are a helpful assistant. You should think step-by-step and put your final answer within \\boxed{{}}.",},
                {"role":"user","content":f"Solve the problem: {question}{speed_prompt}"},
            ]

        return prompt

    if "QwQ" in model or "Qwen" in model:

        prompt = [
                {"role": "system", "content": f"You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step and put your final answer within \\boxed{{}}.",},
                {"role":"user","content":f"Solve the problem: {question}{speed_prompt}"},
        ]
            
        return prompt

    if model == "LLAMA70B" or model == "LLAMA8B":
        prompt = [
            {"role": "system", "content": f"You are a helpful and harmless assistant. You should think step-by-step and put your final answer within \\boxed{{}}.",},
            {"role":"user","content":f'''Please think step-by-step and put your final answer within \\boxed{{}}.\nQuestion:{question}{speed_prompt}'''},
        ]
        return prompt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--K", type=int, default=64)  # model path
    parser.add_argument("--dataset", type=str, default='math_train_hard')  # data path
    parser.add_argument("--model", type=str, default="QwQ")  # output dir
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--speed", type=str, default="normal")
    parser.add_argument("--n_gpus", type=int, default=4)
    parser.add_argument("--save_output", type=bool, default=True)
    
    return parser.parse_args()


dataset_paths = {
    "math_test": "./data/dataset/math_test.json",
    "math_train_hard": "./data/dataset/math_train_hard.json",
    "math_train": "./data/dataset/math_train.json",
    "aime": "./data/dataset/aime.json",
    "gsm8k": "./data/dataset/gsm8k.json",
    "gaokao": "./data/dataset/gaokao.json"
    # "gpqa": "./data/dataset/gpqa_main.json",
}

model_names = {
    "QwQ": "Qwen/QwQ-32B-Preview",
    "Qwen7B": "Qwen/Qwen2.5-Math-7B-Instruct",
    "LLAMA70B": "meta-llama/Llama-3.1-70B-Instruct",
    "LLAMA8B": "meta-llama/Llama-3.1-8B-Instruct",
    "Marco":"AIDC-AI/Marco-o1",
    "Marco-sft-shortest-iter0": "./saves/math/Marco-sft-shortest-iter0",
    "Marco-dpo-iter0":"./saves/math/Marco-dpo-iter0",
    "Marco-loft-iter0":"./saves/math/Marco-loft-iter0-alpha_5-lb_-1"
}


# max_solution_tokens_dict = {
#     "Marco": 8192,
#     "Marco-sft-shortest-iter0": 8192,
#     "QwQ": 8192,
#     "LLAMA70B": 4096,
#     "LLAMA8B": 4096,
#     "Qwen7B": 3072
# }

args = parse_args()
speed = args.speed
K = args.K
model = args.model
dataset = args.dataset
num_samples = args.num_samples #10000
n_gpus = args.n_gpus
max_tokens = 8192 #max_solution_tokens_dict[model]
input_path = dataset_paths[dataset]
model_path = args.model_path
save_output = args.save_output


device = "cuda"
print("INFERENCE:",K,model,dataset,"num:",num_samples)
llm = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")#LLM(model=model_path,tensor_parallel_size=n_gpus,dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained(model_path)
data = json.load(open(input_path,"r"))
print("all num problem:",len(data))

random.seed(42)
random.shuffle(data)

data = data[0:num_samples]
print("used num problem:",len(data))
initial_data = [copy.deepcopy(item) for item in data]
data = [copy.deepcopy(item) for item in data for _ in range(K)]
prompts = [prepare_prompts_for_solution(item['problem'], model, speed=speed) for item in data]
prompts = tokenizer.apply_chat_template(prompts,add_generation_prompt=True,tokenize=False)
print(prompts[0])

import time

# 初始化总生成时间
total_generation_time = 0.0

# 生成答案并统计时间
for prompt in tqdm(prompts):
    encodings = tokenizer([prompt], return_tensors="pt", max_length=2048, truncation=True).to(device)

    # 记录生成的开始时间
    generation_start = time.time()
    llm.generate(**encodings, max_new_tokens=8192, do_sample=False, temperature=None)
    # 记录生成的结束时间
    generation_end = time.time()

    # 计算并累加生成时间
    generation_time = generation_end - generation_start
    total_generation_time += generation_time

    # 输出当前 prompt 的生成时间
    # print(f"Prompt generation time: {generation_time:.2f} seconds")

# 输出所有 prompts 的总生成时间
print(model)
print(f"\nTotal generation time for all prompts: {total_generation_time:.2f} seconds")


# outputs = llm.generate(prompts, sampling_params)
# results = []

# output_solutions = []
# for output in outputs:
#     context = output.prompt
#     generated = output.outputs[0].text
#     output_solutions.append(generated)

# count = 0
# correct = 0

# for i in range(len(data)):
#     item = data[i]
#     answer = extract_answer(output_solutions[i], model_path)
#     if grade_answer(answer, item['ground_truth_answer']):
#         correct += 1
#     count += 1

#     item['solution'] = output_solutions[i]

#     item['answer'] = answer
#     results.append(item)

# print(results[0])
# print(results[0].keys)

# if save_output:
#     print("[Saved]")
#     json.dump(results,open(output_path,"w"))
# else:
#     print("[Not saved]")
