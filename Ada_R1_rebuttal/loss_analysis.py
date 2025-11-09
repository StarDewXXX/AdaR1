import json, os, sys
import math
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', trust_remote_code=True)
merge_long_loss_path = "/home/user1/projects/O1-Pruner-test/Ada_R1_rebuttal/loss_output/long_0.8_short_0.2_long_cot_loss.json"
merge_short_loss_path = "/home/user1/projects/O1-Pruner-test/Ada_R1_rebuttal/loss_output/long_0.8_short_0.2_short_cot_loss.json"

short_model_short_path = '/home/user1/projects/O1-Pruner-test/Ada_R1_rebuttal/loss_output/Deepseek-Qwen-7B-Short-COT_short_cot_loss.json'
long_model_long_path = '/home/user1/projects/O1-Pruner-test/Ada_R1_rebuttal/loss_output/DeepSeek-R1-Distill-Qwen-7B_long_cot_loss.json'


def load_loss_data_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

merge_long_data = load_loss_data_json(merge_long_loss_path)
merge_short_data = load_loss_data_json(merge_short_loss_path)
short_model_short_data = load_loss_data_json(short_model_short_path)
long_model_long_data = load_loss_data_json(long_model_long_path)

def calculate(a, b):
    # print("log prob:",a,b)
    return (a - b) / b 
 
data_nums = len(merge_long_data)
Long_list = []
short_list = []
all_list = []
L0_loss_list, S0_loss_list, L1_loss_list, S1_loss_list = [], [], [], []
for i, item in enumerate(merge_long_data):
    # print(item)  # answer_log_probs total_log_prob question solution_steps
    assert item['question'] == merge_short_data[i]['question'] == short_model_short_data[i]['question'] == long_model_long_data[i]['question'], "Question mismatch between datasets"
    L0 = long_model_long_data[i]['total_log_prob']
    S0 = short_model_short_data[i]['total_log_prob']
    L1 = merge_long_data[i]['total_log_prob']
    S1 = merge_short_data[i]['total_log_prob']

    long_data_lens = tokenizer(long_model_long_data[i]['solution_steps'], return_tensors='pt').input_ids.shape[1]
    short_data_lens = tokenizer(short_model_short_data[i]['solution_steps'], return_tensors='pt').input_ids.shape[1]
    # print(long_data_lens, short_data_lens)
    # sys.exit()

    # Long_ = calculate(L0, L1)
    # Short_ = calculate(S0, S1)

    L0_loss = (-L0) / long_data_lens
    S0_loss = (-S0) / short_data_lens
    L1_loss = (-L1) / long_data_lens
    S1_loss = (-S1) / short_data_lens
    # print('l0_loss:', L0_loss, 's0_loss:', S0_loss, 'l1_loss:', L1_loss, 's1_loss:', S1_loss)
    L0_loss_list.append(L0_loss)
    S0_loss_list.append(S0_loss)
    L1_loss_list.append(L1_loss)
    S1_loss_list.append(S1_loss)

    Long_ = calculate(L0_loss, L1_loss)
    Short_ = calculate(S0_loss, S1_loss)

    Long_list.append(Long_)
    short_list.append(Short_)
    all_list.append(Short_)
    all_list.append(Long_)

print(f"Average Long COT Loss Change: {sum(Long_list) / data_nums:.4f}")
print(f"Average Short COT Loss Change: {sum(short_list) / data_nums:.4f}")
print(f"Average All Change:{sum(short_list) / (2*data_nums):.4f}")
     
print(f"Average Long COT Loss: {sum(L1_loss_list) / data_nums:.4f}")
print(f"Average Short COT Loss: {sum(S1_loss_list) / data_nums:.4f}")
print(f"Average Long COT Loss before merge: {sum(L0_loss_list) / data_nums:.4f}")
print(f"Average Short COT Loss before merge: {sum(S0_loss_list) / data_nums:.4f}")

