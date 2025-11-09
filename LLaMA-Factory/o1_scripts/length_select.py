import torch
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import random
from utils import extract_answer, get_example
from grader import grade_answer
from collections import defaultdict
from datasets import load_dataset
import os
import argparse
import json
import time
import os
import sys
import re
import random
import numpy as np

random.seed(42)

def find_all_valid_equivalence_classes(objects, equivalence_func):
    # 字典用来记录等价类代表元素和等价类的所有元素
    equivalence_classes = {}

    for obj in objects:
        if obj == "None":
            continue
        found_class = False
        for rep in equivalence_classes.keys():
            # 判断当前对象是否和已有代表元素等价
            if obj == rep:
                equivalence_classes[rep].append(obj)
                found_class = True
                break
            elif equivalence_func(obj, rep):
                equivalence_classes[rep].append(obj)
                found_class = True
                break
        # 如果没有找到等价的代表元素，则将当前对象作为新的等价类的代表
        if not found_class:
            equivalence_classes[obj] = [obj]

    # 按等价类大小排序输出
    sorted_classes = sorted(equivalence_classes.values(), key=len, reverse=True)
    return sorted_classes

def to_sft_data(item):
    new_item = {
        "messages":[
            {"role":"system", "content":f"You are a helpful assistant. You should think step-by-step and put your final answer within \\boxed{{}}."},
            {"role":"user", "content":item['problem']},
            {"role":"assistant", "content":item['solution']}
        ],
        # "weight": 1
    }
    return new_item

def to_loft_data(item):
    new_item = {
        "messages":[
            {"role":"system", "content":f"You are a helpful assistant. You should think step-by-step and put your final answer within \\boxed{{}}."},
            {"role":"user", "content":item['problem']},
            {"role":"assistant", "content":item['solution']}
        ],
        "weight": item['weight']
    }
    return new_item

def to_dpo_data(item):
    new_item = {
        "instruction": item['problem'],
        "chosen": item['chosen'],
        "rejected":item['rejected']
    }
    return new_item


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=64)  # model path
    parser.add_argument("--model", type=str, default="QwQ")  # output dir
    parser.add_argument("--model_id", type=str, default="")
    parser.add_argument("--file_name", type=str, default="None")
    parser.add_argument("--dataset_type", type=str, default="sft")
    return parser.parse_args()

model_names = {
    "QwQ": "Qwen/QwQ-32B-Preview",
    "Qwen7B": "Qwen/Qwen2.5-Math-7B-Instruct",
    "LLAMA70B": "meta-llama/Llama-3.1-70B-Instruct",
    "LLAMA8B": "meta-llama/Llama-3.1-8B-Instruct",
    "Marco":"AIDC-AI/Marco-o1"
}

args = parse_args()
model = args.model
model_path = model_names[model]
tokenizer = AutoTokenizer.from_pretrained(model_path)

file_name = args.file_name
os.makedirs(f"./data/model_evalution/{file_name}", exist_ok=True)
input_path =  f"./data/model_generated/{file_name}.json"
output_path = f"./data/model_evalution/{file_name}/majority.json"
model_id = args.model_id
if len(model_id) == 0:
    model_id = model
K = args.K
dataset_type = args.dataset_type
K_values = [K]
data = json.load(open(input_path,"r"))
print("num data:",len(data))

def generate_dpo_dataset(data):
    num_problems = len(data) // K

    output_infos = []

    selected_items = []

    solvable = 0
    correct = 0
    high_conf = 0
    correct_high_conf = 0
    total_tokens = 0
    total_min_tokens = 0

    for problem_index in tqdm(range(num_problems)):

        items = data[problem_index*(K):(problem_index+1)*(K)] 

        problem = items[0]['problem']
        
        real_answer = items[0]['ground_truth_answer']
        

        solutions = [item['solution'] for item in items]
        for solution in solutions:
            num_tokens = len(tokenizer(solution)['input_ids'])
            total_tokens += num_tokens

        answers = [item['answer'] for item in items]

        correct_solutions = []
        wrong_solutions = []

        for answer, solution in zip(answers, solutions):
            if answer != "None":
                if grade_answer(answer ,real_answer):
                    correct_solutions.append(solution)
                else:
                    wrong_solutions.append(solution)
            else:
                wrong_solutions.append(solution)
        
        if len(wrong_solutions) >= 1 and len(correct_solutions) >= 1:
            correct_lengths = []
            for solution in correct_solutions:
                num_tokens = len(tokenizer(solution)['input_ids'])
                correct_lengths.append(num_tokens)
            min_index = correct_lengths.index(min(correct_lengths))

            correct_lengths[min_index] = float('inf')
            second_min_index = correct_lengths.index(min(correct_lengths))


            wrong_lengths = []
            for solution in wrong_solutions:
                num_tokens = len(tokenizer(solution)['input_ids'])
                wrong_lengths.append(num_tokens)
            max_index = wrong_lengths.index(max(wrong_lengths))

            wrong_lengths[max_index] = float('-100.0')
            second_max_index = wrong_lengths.index(max(wrong_lengths))

            selected_items.append(
                {
                    "problem":problem,
                    "chosen":correct_solutions[min_index],
                    "rejected": wrong_solutions[max_index]
                }
            )

            selected_items.append(
                {
                    "problem":problem,
                    "chosen":correct_solutions[second_min_index],
                    "rejected": wrong_solutions[max_index]
                }
            )

            selected_items.append(
                {
                    "problem":problem,
                    "chosen":correct_solutions[min_index],
                    "rejected": wrong_solutions[second_max_index]
                }
            )

            selected_items.append(
                {
                    "problem":problem,
                    "chosen":correct_solutions[second_min_index],
                    "rejected": wrong_solutions[second_max_index]
                }
            )

    dataset_data = [to_dpo_data(item) for item in selected_items]

    save_dir = f"../LLaMA-Factory/data/my_dataset/{model}-iter0-MATH-train-K-16-dpo"
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{save_dir}/raw.json"
    with open(filename,"w") as f:
        json.dump(dataset_data,f)
    dataset = load_dataset("json",data_files=filename)
    dataset.save_to_disk(f"{save_dir}")
    print(dataset)
    print(save_dir)


def generate_sft_dataset(data):
    num_problems = len(data) // K

    output_infos = []

    selected_items = []

    solvable = 0
    correct = 0
    high_conf = 0
    correct_high_conf = 0
    total_tokens = 0
    total_min_tokens = 0

    for problem_index in tqdm(range(num_problems)):

        items = data[problem_index*(K):(problem_index+1)*(K)] 
        random.shuffle(items)

        problem = items[0]['problem']
        
        real_answer = items[0]['ground_truth_answer']
        

        solutions = [item['solution'] for item in items]
        for solution in solutions:
            num_tokens = len(tokenizer(solution)['input_ids'])
            total_tokens += num_tokens

        answers = [item['answer'] for item in items]

        correct_solutions = []
        for answer, solution in zip(answers, solutions):
            if answer != "None":
                if grade_answer(answer ,real_answer):
                    correct_solutions.append(solution)
        
        if len(correct_solutions) >= 1:

            lengths = []
            for solution in correct_solutions:
                num_tokens = len(tokenizer(solution)['input_ids'])
                lengths.append(num_tokens)
            

            min_index = lengths.index(min(lengths))
            max_index = lengths.index(max(lengths))
            selected_item = {
                "problem": problem,
                "solution":correct_solutions[min_index],
            }
            total_min_tokens += lengths[min_index]
            selected_items.append(selected_item)
        
    print(f"correct ratio:{correct/num_problems} avg_tokens:{total_tokens/(K*num_problems)} selected_avg_tokens:{total_min_tokens/len(selected_items)}")

    dataset_data = [to_sft_data(item) for item in selected_items]

    save_dir = f"../LLaMA-Factory/data/my_dataset/{model}-iter0-MATH-train-K-16-shortest"
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{save_dir}/raw.json"
    with open(filename,"w") as f:
        json.dump(dataset_data,f)
    dataset = load_dataset("json",data_files=filename)
    dataset.save_to_disk(f"{save_dir}")
    print(dataset)


def generate_loft_dataset(data):

    def get_token_length(solution):
        return len(tokenizer(solution)['input_ids'])
    num_problems = len(data) // K

    output_infos = []

    selected_items = []

    solvable = 0
    correct = 0
    high_conf = 0
    correct_high_conf = 0
    total_tokens = 0
    total_min_tokens = 0

    count_per_problem = 2
    alpha = 3 #5

    lower_bound = -2
    upper_bound = 4

    weights = []

    for problem_index in tqdm(range(num_problems)):

        items = data[problem_index*(K):(problem_index+1)*(K)] 
        random.shuffle(items)

        problem = items[0]['problem']
        
        real_answer = items[0]['ground_truth_answer']
        
        answers = [item['answer'] for item in items]
        solutions = [item['solution'] for item in items]
        correctness = [grade_answer(answer, real_answer) for answer in answers]

        avg_length = sum([get_token_length(solution) for solution in solutions]) / len(solutions)
        avg_acc = sum([int(c) for c in correctness]) / len(correctness)

        for index in range(0, count_per_problem):#random.sample(range(0, len(solutions)), count_per_problem):
            solution = solutions[index]
            length = get_token_length(solution)
            length_term = (avg_length - length) / length
            acc_term = alpha * (int(correctness[index]) - avg_acc)
            weight = length_term + acc_term
            if weight <= lower_bound:
                weight = lower_bound
            if weight > upper_bound:
                weight = upper_bound
            print(f"length:{length} acc:{correctness[index]}, avg_length:{avg_length} avg_acc:{avg_acc}, weight:{weight}")
            selected_items.append(
                {
                    "problem": problem,
                    "solution": solution,
                    "weight": float(weight)
                }
            )
            weights.append(weight)
        
    mean_weight = np.mean(weights)
    std_weight = np.std(weights)

    for item in selected_items:
        unnormlized_weight = item["weight"]
        item["weight"] = (item["weight"] - mean_weight) / std_weight
        # print("unnormlized_weight:",unnormlized_weight,"after:",item["weight"])
    
    print("mean_weight:",mean_weight)
    print("std_weight",std_weight)
            

    print("num selected:",len(selected_items))

    dataset_data = [to_loft_data(item) for item in selected_items]

    save_dir = f"../LLaMA-Factory/data/my_dataset/{model_id}-iter0-MATH-train-K-16-loft-alpha-{alpha}-k-{count_per_problem}"
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{save_dir}/raw.json"
    with open(filename,"w") as f:
        json.dump(dataset_data,f)
    dataset = load_dataset("json",data_files=filename)
    dataset.save_to_disk(f"{save_dir}")
    print(dataset)
    print(dataset['train'][0])
    print(save_dir)

generate_loft_dataset(data)
# generate_dpo_dataset(data)
# generate_sft_dataset(data)