from vllm import LLM, SamplingParams
import torch
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import random
import math

model_path = "AIDC-AI/Marco-o1"
llm = LLM(model=model_path,tensor_parallel_size=2,dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained(model_path)
sampling_params = SamplingParams(temperature=0,top_p=0.1,max_tokens=1,prompt_logprobs=20)

raw_chats = [[{"role":"user","content":"How are you?"},{"role":"assistant","content":"I am fine and thank you."}]]

chats_tokenzied = tokenizer.apply_chat_template(raw_chats, tokenize=True, continue_final_message=True)
chats = tokenizer.apply_chat_template(raw_chats, tokenize=False, continue_final_message=True)
prompts = [chat[0:1] for chat in raw_chats]
print(prompts)
prompts_tokenzied = tokenizer.apply_chat_template(prompts, tokenize=True, add_generation_prompt=True)
responses_tokenzied = [chat[len(prompt):] for prompt, chat in zip(prompts_tokenzied, chats_tokenzied)]
print(responses_tokenzied)
outputs = llm.generate(chats, sampling_params)
# print(outputs[0])

prompt_logprobs = outputs[0].prompt_logprobs[len(prompts_tokenzied[0]):]
response_tokenzied = responses_tokenzied[0]
print(prompt_logprobs[0])
sum_logprobs = [[probs[label]] for label, probs in zip(response_tokenzied[1:],prompt_logprobs[:-1])]
print(sum_logprobs)
print(len(prompt_logprobs))
print(len(chats_tokenzied[0]))