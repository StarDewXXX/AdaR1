import torch
from thop import profile
import argparse
import json
import time
import os
import sys
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")
# import logging

# logging.getLogger().setLevel(logging.WARNING)  # 设置日志级别为 WARNING 或更高

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default='')  # data path
    parser.add_argument("--model", type=str, default="QwQ")  # output dir
    parser.add_argument("--mode", type=str, default="normal")  # output dir
    # parser.add_argument("--m", type=str, default="QwQ") 
    
    return parser.parse_args()

args = parse_args()
model_name = args.model
mode = args.mode


model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

datasets = [
    "math_test",
    "gsm8k",
    "gaokao"
]

for dataset in datasets:
    input_file = f"./data/model_generated/{args.input_file}_{dataset}_8192_{mode}_K-1.json"

    data = json.load(open(input_file,"r"))
    print(list(data[0].keys()))

    device = "cuda:3"
    model.to(device)


    total_flops = 0
    count = 0

    for index in tqdm(range(len(data))):
        item = data[index]
        problem = item['problem']
        solution = item['solution']
        chat = [
            {"role": "system", "content": f"You are a helpful assistant. You should think step-by-step and put your final answer within \\boxed{{}}."},
            {"role":"user","content":f"Solve the problem: {problem}"},
            {"role":"assistant","content":solution}
        ]
        # print(chat)
        input_str = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        inputs = tokenizer(input_str, return_tensors="pt").to(device)
        # print(inputs)

        flops, params = profile(model, inputs=(inputs['input_ids'],inputs['attention_mask']),verbose=False)
        flops = flops / 1E9
        total_flops += flops
        count += 1
    print("input file:", input_file)
    print("avg flops:",total_flops / count)
    print("-"*100)

