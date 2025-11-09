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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from utils import load_eval_data
from deepscaler.rewards.math_utils.utils import extract_answer, grade_answer_sympy as grade_answer

random.seed(0)


import numpy as np

def normalize_gain(lst, baseline=0):
    arr = np.array(lst, dtype=np.float32)
    std = np.std(arr)
    print("std:", std) 
    return (arr-baseline) / std

def transform(arr):
    # if element > 0, add 1; if element < 0, minus 1
    return [element+1 if element >= 0 else element-1 for element in arr]

def format_dataset(input_path, output_path):

    def format(example):
        example["messages"] = [{"role":"user","content":f"{example['problem']} Let's think step by step and output the final answer within \\boxed{{}}."},{"role":"assistant","content":example['solution']}]
        return example

    ds = load_dataset("json", data_files=input_path)

    columns_to_remove = ['ground_truth_solution','ground_truth_answer', 'pre_generated_steps', 'pre_generated_answer', 'pre_generated_verifier_score']
    for column in columns_to_remove:
        if column in list(ds['train'][0].keys()):
            ds = ds.remove_columns([column])
            print("remove column:", column)

    ds = ds.map(format)
    ds = ds.remove_columns(['problem', 'solution'])

    print(ds)
    print(ds['train'][0])
    print(ds['train'][1])
    ds.save_to_disk(output_path)

def format_pairwise_sample(chosen_item, rejected_item, weight):
    prompt = chosen_item['prompt']
    chosen_response = chosen_item['solution']
    rejected_response = rejected_item['solution']

    sample = {
        "instruction": prompt,
        "chosen": chosen_response,
        "rejected": rejected_response
    }

    if weight != None:
        sample['weight'] = weight
    
    return sample

def format_sft_sample(item):
    prompt = item['prompt']
    response = item['solution']
    sample = {
        "messages":[
            {"role":"user","content":prompt},
            {"role":"assistant","content":response}
        ]
    }
    return sample

def filter_group(group, ground_truth_answer, filter_type):
    if filter_type == "wrong":
        return [item for item in group if grade_answer(extract_answer(item['solution']), ground_truth_answer) == False]
    if filter_type == "correct":
        # for item in group:
        #     answer = extract_answer(item['solution'])
        #     print(answer,ground_truth_answer,grade_answer(answer, ground_truth_answer))
        # print("-"*10)
        return [item for item in group if grade_answer(extract_answer(item['solution']), ground_truth_answer) == True]
    
def construct_pairwise_json_data(data, max_pairs, with_weight=False, bi_level=False):
    output_data = []

    for item in data:
        gain = item['gain']
        ground_truth_answer = item['ground_truth_answer']

        if gain <= 0:
            chosen_group = item['short_group']
            rejected_group = item['long_group']
        
        if gain > 0:
            chosen_group = item['long_group']
            rejected_group = item['short_group']
        
        correct_chosen_group = filter_group(chosen_group, ground_truth_answer, "correct")

        if len(correct_chosen_group) != 0:
            chosen_group = correct_chosen_group
        
        inner_group_samples = []
        if bi_level:

            shorest_item, longest_item = None, None

            correct_chosen_group = filter_group(chosen_group, ground_truth_answer, "correct")
            wrong_chosen_group = filter_group(chosen_group, ground_truth_answer, "wrong")
            
            correct_chosen_group_lengths = [len(tokenizer(item['solution'])['input_ids']) for item in correct_chosen_group]
            wrong_chosen_group_lengths = [len(tokenizer(item['solution'])['input_ids']) for item in wrong_chosen_group]

            M = 2
            if len(correct_chosen_group) != 0 and len(wrong_chosen_group) != 0:
                shortest_idx = correct_chosen_group_lengths.index(min(correct_chosen_group_lengths))
                longest_indices = sorted(range(len(wrong_chosen_group_lengths)), key=lambda i: -wrong_chosen_group_lengths[i])[:M]
                shorest_item = correct_chosen_group[shortest_idx]
                longest_item = [wrong_chosen_group[i] for i in longest_indices]

            elif len(correct_chosen_group) != 0 and len(wrong_chosen_group) == 0:
                sorted_indices = sorted(range(len(correct_chosen_group_lengths)), key=lambda i: correct_chosen_group_lengths[i])
                shorest_item = correct_chosen_group[sorted_indices[0]]
                longest_item = [correct_chosen_group[i] for i in sorted_indices[-M:]]

            elif len(correct_chosen_group) == 0 and len(wrong_chosen_group) != 0:
                sorted_indices = sorted(range(len(wrong_chosen_group_lengths)), key=lambda i: wrong_chosen_group_lengths[i])
                shorest_item = wrong_chosen_group[sorted_indices[0]]
                longest_item = [wrong_chosen_group[i] for i in sorted_indices[-M:]]

            for long_item in longest_item:
                inner_group_samples.append(format_pairwise_sample(shorest_item, long_item, 1))
                # print("[shorest_item]:",shorest_item)
                # print("[longest_item]:",long_item)
                # print("-"*100)
                # input("continue?")
            


        
        wrong_rejected_group = filter_group(rejected_group, ground_truth_answer, "wrong")
        if len(wrong_rejected_group) != 0:
            rejected_group = wrong_rejected_group
        
        all_samples = []
        for chosen_item in chosen_group:
            for rejected_item in rejected_group:
                if with_weight:
                    weight = abs(gain)
                    all_samples.append(format_pairwise_sample(chosen_item, rejected_item, 1))
                else:
                    all_samples.append(format_pairwise_sample(chosen_item, rejected_item, 1))

        random.shuffle(all_samples)
        all_samples = all_samples[0:max_pairs]
        output_data += all_samples # inter-group
        output_data += inner_group_samples # inner-group
    return output_data

def json_to_dataset(data, output_path):
    # save json as temp file
    with open("temp.json", "w") as f:    
        json.dump(data, f)

    ds = load_dataset("json", data_files="temp.json")
    print(ds)
    print(ds['train'][0])
    if "weight" in ds['train'][0].keys():
        print(ds['train'][0]['weight'])
        print(ds['train'][1]['weight'])
        print(ds['train'][2]['weight'])
        print(ds['train'][3]['weight'])
        print(ds['train'][4]['weight'])
    # print(ds['train'][1])
    ds.save_to_disk(output_path)
    print("dataset saved to", output_path)
    return ds

if __name__ == "__main__":
    model_path = "Qwen/Qwen2.5-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    long_cot_data_path = "model_eval/DeepSeek-R1-Distill-Qwen-7B/mix_mathematic_problems.json"
    short_cot_data_path = "model_eval/Deepseek-Qwen-7B-Short-COT/mix_mathematic_problems.json"

    long_cot_data = load_eval_data(long_cot_data_path)
    short_cot_data = load_eval_data(short_cot_data_path)

    K = len(long_cot_data[0]['responses'])
    print("K:", K)

    assert len(long_cot_data) == len(short_cot_data)

    print("num problems:",len(long_cot_data))
    # sys.exit()
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

    max_length_inc_ratio = 10

    acc_counters = [0,0,0] # >0 =0 <0
    long_lengths = []
    accuracy_diffs = []
    for group_index in range(len(long_cot_data)):
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
            

        all_gain.append(gain)
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

        
        if long_avg_length > 4000 and gain < 0:
            print("long_accuracy:",long_accuracy)
            print("short_accuracy:",short_accuracy)
            print("long_avg_length:",long_avg_length)
            print("short_avg_length:",short_avg_length)
            print("relative_accuracy_gain:",relative_accuracy_gain)
            print("relative_length_increnment:",relative_length_increnment)
            print("gain:",gain)
            print("-"*20)


    print("acc_counters:", [a / valid_count for a in acc_counters])

    np.set_printoptions(suppress=True)
    gain_avg = sum(all_gain) / len(all_gain)
    gain_abs_avg = sum([e if e>=0 else -e for e in all_gain]) / len(all_gain)
    print("gain_avg:", gain_avg)
    print("gain abs avg:", gain_abs_avg)
    print("gain max:", max(all_gain))
    print("gain min:", min(all_gain))

    print("[-] negative gain:",negative / (valid_count))
    print("[+] postive gain:",postive / (valid_count))

    print("total_acc_long:", total_acc_long / valid_count, "total_length_long:", total_length_long / valid_count)
    print("total_acc_short:", total_acc_short / valid_count, "total_length_short:", total_length_short / valid_count)
    print("total_acc_optimal:", total_acc_optimal / valid_count, "total_length_optimal:", total_length_optimal / valid_count)


    # all_gain_transformed = normalize_gain(all_gain)
    all_gain_transformed = [e/gain_abs_avg for e in all_gain]
    all_gain_transformed = transform(all_gain_transformed)

    # # bilevel (group level + instance level)
    output_path = "./ds-1.5b_dpo_bilevel_M1-4-M2-2"
    json_data = construct_pairwise_json_data(selected_data, 4, with_weight=False, bi_level=True)
    json_to_dataset(json_data, output_path)