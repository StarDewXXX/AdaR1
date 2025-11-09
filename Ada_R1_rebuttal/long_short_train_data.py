import json
# from utils import extract_answer
# from grader import grade_answer
from datasets import load_from_disk
from datasets import load_dataset
import random
from transformers import AutoTokenizer
import numpy as np
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import sys
import os
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from utils import load_eval_data
from deepscaler.rewards.math_utils.utils import extract_answer, grade_answer_sympy as grade_answer
from constrcut_adaptive_dataset import construct_sft_json_data, json_to_dataset
random.seed(0)
negative = 0
postive = 0

long_acc_random = 0
short_acc_random = 0

valid_count = 0

all_gain = []

selected_data = []

total_acc_long = 0
total_length_long = 0

total_acc_short = 0
total_length_short = 0

total_acc_optimal = 0
total_length_optimal = 0

# positive_coefficent = 10
# negative_coefficent = 1

max_length_inc_ratio = 10

acc_counters = [0,0,0] # >0 =0 <0
long_lengths = []
accuracy_diffs = []
model_path = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
import numpy as np
long_cot_data_path = "model_eval/DeepSeek-R1-Distill-Qwen-7B/mix_mathematic_problems.json"
short_cot_data_path = "model_eval/Deepseek-Qwen-7B-Short-COT/mix_mathematic_problems.json"
selected_data = []
long_cot_data = load_eval_data(long_cot_data_path)
short_cot_data = load_eval_data(short_cot_data_path)
K = len(long_cot_data[0]['responses'])
print("K:", K)

assert len(long_cot_data) == len(short_cot_data)

print("num problems:",len(long_cot_data))
# sys.exit()
nums = random.sample(range(2500), 1000)
# for group_index in range(len(long_cot_data)):
# for group_index in nums:
print('random sampled nums:', nums)
for group_index in tqdm(nums, desc="Processing groups"):
# for group_index in range(50):
# 
    long_group = long_cot_data[group_index]
    short_group = short_cot_data[group_index]

    # calculate correctness
    ground_truth_answer = long_group['reward_model']['ground_truth']

    # skip multiple choice questions
    if ground_truth_answer in ["A", "B", "C", "D", "E", "F", "\\text{A}", "\\text{B}", "\\text{C}", "\\text{D}", "\\text{E}", "\\text{F}","\\boxed{A}", "\\boxed{B}", "\\boxed{C}", "\\boxed{D}", "\\boxed{E}", "\\boxed{F}"]:
        continue
    if ground_truth_answer =="None" or ground_truth_answer == "":
        continue

    long_answers = [extract_answer(solution) for solution in long_group['responses']]
    short_answers = [extract_answer(solution) for solution in short_group['responses']]

    long_correctness = [grade_answer(answer, ground_truth_answer) for answer in long_answers]
    short_correctness = [grade_answer(answer, ground_truth_answer) for answer in short_answers]

    long_acc_random += long_correctness[0]
    short_acc_random += short_correctness[0]

    # calculate lengths
    long_solutions = [solution for solution in long_group['responses']]
    short_solutions = [solution for solution in short_group['responses']]


    long_solution_lengths = [len(tokenizer(solution)['input_ids']) for solution in long_solutions]
    short_solution_lengths = [len(tokenizer(solution)['input_ids']) for solution in short_solutions]

    long_accuracy = sum(long_correctness) / len(long_correctness)
    short_accuracy = sum(short_correctness) / len(short_correctness)

    long_avg_length = sum(long_solution_lengths) / len(long_solution_lengths)
    short_avg_length = sum(short_solution_lengths) / len(short_solution_lengths)

    
    relative_accuracy_gain = long_accuracy - short_accuracy - 1/(2*K) #/ short_accuracy if short_accuracy != 0 else (long_accuracy - 1/K) / (1/K)
    relative_length_increnment = (long_avg_length - short_avg_length) / short_avg_length
    
    if relative_accuracy_gain > 0:
        gain = relative_accuracy_gain / relative_length_increnment
    else:
        gain = relative_accuracy_gain * (relative_length_increnment/max_length_inc_ratio)

    # a special case
    if long_accuracy == 0 and short_accuracy == 0:
        continue
    
    valid_count += 1
    
    acc_diff = long_accuracy - short_accuracy
    acc_counters[0] += 1 if acc_diff > 0 else 0
    acc_counters[1] += 1 if acc_diff == 0 else 0
    acc_counters[2] += 1 if acc_diff < 0 else 0

    long_lengths.append(long_avg_length)
    accuracy_diffs.append(acc_diff)



    total_acc_long += long_correctness[0]
    total_length_long += long_solution_lengths[0]

    total_acc_short += short_correctness[0]
    total_length_short += short_solution_lengths[0]

    if gain > 0:
        postive += 1
        total_acc_optimal += long_correctness[0]
        total_length_optimal += long_solution_lengths[0]

    if gain <= 0:
        negative += 1
        total_acc_optimal += short_correctness[0]
        total_length_optimal += short_solution_lengths[0]
        

    # all_gain.append(gain)
    prompt = long_group['prompt'][0]['content']
    formatted_long_group = [{"prompt":prompt, "solution":solution, "ground_truth_answer":ground_truth_answer} for solution in long_group['responses']]
    formatted_short_group = [{"prompt":prompt, "solution":solution, "ground_truth_answer":ground_truth_answer} for solution in short_group['responses']]
    selected_data.append({
        "prompt": prompt,
        "ground_truth_answer": ground_truth_answer,
        "long_group": formatted_long_group,
        "short_group": formatted_short_group,
        "gain": gain
    })

output_path = "../../O1-Pruner-test/data/my_dataset/ds-7b_rebuttal_long-short-mixed-sft"
json_data = construct_sft_json_data(selected_data, 4, optimal='rebuttal_equal')
print('====='*20)
print("len saved data:", len(json_data))

json_to_dataset(json_data, output_path)