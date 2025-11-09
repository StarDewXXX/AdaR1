import json
import random
from datasets import load_dataset
from deepscaler.rewards.math_utils.utils import extract_answer, grade_answer_sympy as grade_answer

# math train
original_data_path = "/home/user1/projects/verl/length_control/data/dataset/math_train.json"
original_data = json.load(open(original_data_path, 'r'))
new_data = []
print(original_data[0])
random.shuffle(original_data)
print(original_data[0])
for index in range(5000):
    item = original_data[index]
    new_data.append({
        "problem": item["problem"],
        "answer": item['ground_truth_answer']
    })

output_data_path = "deepscaler/data/test/math_train.json"
with open(output_data_path, 'w') as f:
    json.dump(new_data, f)

print(len(new_data))

# deepscaler
original_data_path = "deepscaler/data/train/deepscaler.json"
original_data = json.load(open(original_data_path, 'r'))
new_data = []
print(original_data[0])
print(original_data[1])
print(original_data[2])
print("original size:",len(original_data))
random.shuffle(original_data)
for index in range(2000):
    item = original_data[index]
    new_data.append({
        "problem": item["problem"],
        "answer": item['answer']
    })

output_data_path = "deepscaler/data/test/scaler.json"
with open(output_data_path, 'w') as f:
    json.dump(new_data, f)

print("new_data size:",len(new_data))