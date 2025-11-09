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
import matplotlib.pyplot as plt

def get_length(solution):
    return len(tokenizer(solution)['input_ids'])

import numpy as np
from collections import defaultdict

def calculate_accuracy_dynamic_intervals(solutions, correctness, num_intervals=5):
    # 获取每个solution的长度
    lengths = [get_length(solution) for solution in solutions]
    
    # 动态划分区间：按长度的百分位数
    quantiles = np.linspace(0, 1, num_intervals + 1)
    intervals = np.quantile(lengths, quantiles)

    print(intervals)
    
    # 初始化区间统计
    interval_correct = defaultdict(int)
    interval_total = defaultdict(int)

    # 归类到区间
    for length, correct in zip(lengths, correctness):
        for i in range(len(intervals) - 1):
            if intervals[i] < length <= intervals[i + 1]:  # 区间包括右边界
                interval_total[(intervals[i], intervals[i + 1])] += 1
                if correct == 1:  # 正确的答案
                    interval_correct[(intervals[i], intervals[i + 1])] += 1
                break

    # 计算准确率
    interval_accuracy = {}
    for interval, total in interval_total.items():
        correct = interval_correct[interval]
        accuracy = correct / total if total > 0 else 0
        print(total)
        interval_accuracy[interval] = accuracy

    return interval_accuracy



model = "AIDC-AI/Marco-o1"
model_name = "Marco-o1"
K = 512
data_path = "./data/model_generated/Marco_math_train_8192_normal_K-512.json"

# model = "Qwen/QwQ-32B-Preview"
# model_name = "QwQ-32B-Preview"
# K = 512
# data_path = "./data/model_generated/QwQ_math_train_hard_8192_normal_K-512.json"

tokenizer = AutoTokenizer.from_pretrained(model)
os.makedirs(f"./plots/{model}", exist_ok=True)

num_intervals = 4
final_results = [0 for i in range(num_intervals)]
assert str(K) in data_path
data = json.load(open(data_path,"r"))
num_problems = len(data) // K

for index in range(num_problems):
    items = data[index*K:(index+1)*K]
    solutions = [item['solution'] for item in items if item['answer'] != "None"]
    answers = [item['answer'] for item in items if item['answer'] != "None"]
    # solutions = [item['solution'] for item in items]
    # answers = [item['answer'] for item in items]

    real_answer = items[0]['ground_truth_answer']

    correctness = [int(grade_answer(answers[i], real_answer)) for i in range(len(answers))]
    # for i in range(K):
    #     if answers[i] == "None":
    #         correctness[i] = -1
    
    # lengths = [get_length(solution) for solution in solutions]
    accuracy_by_dynamic_intervals = calculate_accuracy_dynamic_intervals(solutions, correctness, num_intervals=num_intervals)
    results = []
    intervals = []
    # 输出结果

    count = 0 
    for interval, accuracy in sorted(accuracy_by_dynamic_intervals.items()):
        intervals.append((interval[0]+interval[1])/2)
        results.append(accuracy)

        final_results[count] += accuracy
        count += 1
    
    print(final_results)
    # x_indices = list(range(num_intervals))
    x_indices = [str(int(x)) for x in intervals]
    plt.bar(x_indices, results, color="skyblue", edgecolor="black")
    plt.xlabel("Token Length")
    plt.ylabel("Accuracy")

    # plt.plot(intervals, results)
    plt.title(f"Problem {index}\n{model_name}")
    plt.savefig(f"./plots/{model}/problem_{index}.png",dpi=100)

    # 清空图像
    plt.clf()

    


print("-"*100)
final_results = [x/num_problems for x in final_results]
x_indices = list(range(num_intervals))
plt.bar(x_indices, final_results, color="skyblue", edgecolor="black")
plt.xlabel("Token Length Interval")
plt.ylabel("Accuracy")

# plt.plot(intervals, results)
plt.title(f"All Problems\n{model_name}")
plt.savefig(f"./plots/{model}/all.png",dpi=100)
print(final_results)



