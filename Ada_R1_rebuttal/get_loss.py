import torch, os, sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
import torch.nn.functional as F
# data_path = "/home/user1/projects/O1-Pruner-test/data/dataset/math_train.json"
long_cot_path = "/home/user1/projects/Light-R1/deepscaler-release/model_eval/DeepSeek-R1-Distill-Qwen-7B/math.json"
short_cot_path = "/home/user1/projects/Light-R1/deepscaler-release/model_eval/Deepseek-Qwen-7B-Short-COT/math.json"


# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# model_name = "/home/user1/projects/verl/length_control/models/Deepseek-Qwen-7B/long_0.8_short_0.2"
model_name = "/home/user1/projects/Light-R1/deepscaler-release/models/Deepseek-Qwen-7B/Deepseek-Qwen-7B-Short-COT"
# model_name = "/home/user1/projects/Light-R1/deepscaler-release/models/Deepseek-Qwen-7B/Deepseek-Qwen-7B-merge-0.8-dpo-beta-0.1-no-ln-bilevel-fulldata-M1-4-M2-2-cosine-lr"
import json


with open(long_cot_path, "r", encoding="utf-8") as f:
    # data = json.load(f)
    long_cot_data = [json.loads(line) for line in f]
with open(short_cot_path, "r", encoding="utf-8") as f:
    short_cot_data = [json.loads(line) for line in f]

tokenizer   = AutoTokenizer.from_pretrained(model_name)
model       = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",           
    torch_dtype=torch.float16     
).eval()
MODEL = model_name.split("/")[-1]  # 获取模型名称
print("Model loaded from:\n", model_name)
short_save_json = f"/home/user1/projects/O1-Pruner-test/Ada_R1_rebuttal/loss_output/{MODEL}_short_cot_loss.json"
short_save_dir = []
long_save_json = f"/home/user1/projects/O1-Pruner-test/Ada_R1_rebuttal/loss_output/{MODEL}_long_cot_loss.json"
long_save_dir = []
print("Processing data...")
sums = 0
for i, item in enumerate(long_cot_data):
    # print(item)
    # sys.exit()
    # question = item['problem']
    # solution_steps = item['ground_truth_solution']
    question = item['prompt']
    assert question == short_cot_data[i]['prompt'], "Question mismatch between long and short COT data"
    question = question[0]['content']
    long_correct_ness = item['correctness']
    short_correct_ness = short_cot_data[i]['correctness']
    if not (long_correct_ness == [1] and short_correct_ness == [1]):
        print(f"Skipping item {i} due to correctness mismatch: long={long_correct_ness}, short={short_correct_ness}")
        continue
    
    solution_steps = item['responses']  # todo:应该有多个  找到long和short同时对的
    short_solution_steps = short_cot_data[i]['responses']
    assert len(solution_steps) == 1, "Expected exactly one solution step in long COT data"
    
    # print(f"Question: {question}")
    # print(f"Solution Steps: {solution_steps}")
    prompt = f'Question: {question}\nAnswer: '  # 模型输入
    
    target  = solution_steps[0] + tokenizer.eos_token          # 监督输出
    short_target = short_solution_steps[0] + tokenizer.eos_token  # 短 COT 的监督输出

    prompt_ids = tokenizer(prompt , return_tensors="pt").to(model.device)["input_ids"]
    target_ids = tokenizer(target , return_tensors="pt").to(model.device)["input_ids"]
    input_ids  = torch.cat([prompt_ids, target_ids], dim=1)        # 形状 [1, seq_len]
    short_target_ids = tokenizer(short_target, return_tensors="pt").to(model.device)["input_ids"]
    short_input_ids = torch.cat([prompt_ids, short_target_ids], dim=1)  # 短 COT 的输入

    # ---------- 前向，拿 logits ----------
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=False)       # outputs.logits: [1, seq_len, vocab]

    logits = outputs.logits                                         # 未做 softmax 的分数

    with torch.no_grad():
        short_outputs = model(input_ids=short_input_ids, use_cache=False)
    short_logits = short_outputs.logits                           # 短 COT 的 logits

    # ---------- (可选) 计算“参考解”路径上的对数概率 ----------
    # 1. 右移对齐：位置 t 的 logits 预测 token_{t+1}
    shift_logits = logits[:, :-1, :]            # [1, seq_len-1, vocab]
    shift_labels = input_ids[:, 1:]             # [1, seq_len-1]

    short_shift_logits = short_logits[:, :-1, :]  # 短 COT 的 logits
    short_shift_labels = short_input_ids[:, 1:]   # 短 COT 的标签

    # 2. 转为对数概率
    log_probs = F.log_softmax(shift_logits, dim=-1)                 # 同形状，取 log P
    short_log_probs = F.log_softmax(short_shift_logits, dim=-1)     # 短 COT 的 log P

    # 3. 取出参考答案路径上每个 token 的 log P
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    # token_log_probs: [1, seq_len-1]，依次对应 EOS 之前每个 token 的 log P
    short_token_log_probs = short_log_probs.gather(-1, short_shift_labels.unsqueeze(-1)).squeeze(-1)

    # 4. 如果只想要“解答”部分的 log P：
    answer_log_probs = token_log_probs[:, prompt_ids.size(1)-1:]    # 从 prompt 末尾开始
    short_answer_log_probs = short_token_log_probs[:, prompt_ids.size(1)-1:]  # 短 COT 的答案 log P
    # 5. 求和得到总的 log P
    total_log_prob = answer_log_probs.sum().item()                  # 形状标量
    answer_log_probs = answer_log_probs.tolist()                     # 转为列表

    short_total_log_prob = short_answer_log_probs.sum().item()      # 短 COT 的总 log P
    short_answer_log_probs = short_answer_log_probs.tolist()  

    long_save_dir.append({
        "question": question,
        "solution_steps": solution_steps,
        "answer_log_probs": answer_log_probs,
        "total_log_prob": total_log_prob
    })
    short_save_dir.append({
        "question": question,
        "solution_steps": short_solution_steps,
        "answer_log_probs": short_answer_log_probs,
        "total_log_prob": short_total_log_prob
    })
    print('item:\{', i, '\}processed')
    sums += 1
    if sums >= 300:
        break
    
# # 保存结果到 JSON 文件
with open(long_save_json, "w", encoding="utf-8") as f:
    json.dump(long_save_dir, f, indent=4, ensure_ascii=False)
print(f"Results saved to {long_save_json}")
with open(short_save_json, "w", encoding="utf-8") as f:
    json.dump(short_save_dir, f, indent=4, ensure_ascii=False)
print(f"Results saved to {short_save_json}")