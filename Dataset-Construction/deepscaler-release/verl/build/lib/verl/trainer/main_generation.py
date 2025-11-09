# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses given a dataset of prompts
"""
import csv
import ray
import numpy as np
import hydra
import os
import time
from tabulate import tabulate
from collections import Counter

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from verl.utils.model import compute_position_id_with_mask

import pandas as pd

from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from deepscaler.rewards.math_reward import deepscaler_reward_fn
from deepscaler.rewards.math_utils.utils import extract_answer


@hydra.main(config_path='config', config_name='generation', version_base=None)
def main(config):
    start_time = time.time()
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    local_path = copy_local_path_from_hdfs(config.model.path)
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)
    # Check if output file already exists
    if os.path.exists(config.data.output_path):
        print(f"Output file {config.data.output_path} already exists. Skipping generation and proceeding to evaluation.")
        if config.data.output_path.endswith('.parquet'):
            dataset = pd.read_parquet(config.data.output_path)
        elif config.data.output_path.endswith('.json'):
            dataset = pd.read_json(config.data.output_path, orient='records', lines=True)
    else:

        if config.rollout.temperature == 0.:
            assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'

        # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
        dataset = pd.read_parquet(config.data.path)
        chat_lst = dataset[config.data.prompt_key].tolist()

        chat_lst = [chat.tolist() for chat in chat_lst]

        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='rollout')
        resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
        wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
        wg.init_model()

        total_samples = len(dataset)
        # real_batch_size = data.batch['input_ids'].shape[0]
        config_batch_size = config.data.batch_size
        dp_size = wg.world_size // config.rollout.tensor_model_parallel_size
        num_batch = (total_samples // config_batch_size) + 1
        if total_samples % config_batch_size == 0:
            num_batch = total_samples // config_batch_size
        output_lst = []  # We'll reshape at the end

        print('len(dataset):', total_samples)
        print('wg.worker_names:', wg.worker_names)

        for batch_idx in range(num_batch):
            print(f'[{batch_idx+1}/{num_batch}] Start to process.')
            batch_chat_lst = chat_lst[batch_idx * config_batch_size:(batch_idx + 1) * config_batch_size]
            
            # Repeat the batch n_samples times
            from pprint import pprint
            repeated_chat_lst = []
            for chat in batch_chat_lst:
                repeated_chat_lst.extend([chat] * config.data.n_samples)  # 这里重复，所以每个生成request n=1
            print('repeated_chat_lst0')
            pprint(repeated_chat_lst[:3])
            repeated_chat_lst = batch_chat_lst * config.data.n_samples  # 负载更均衡
            print('repeated_chat_lst1')
            pprint(repeated_chat_lst[:3])
            
            inputs = tokenizer.apply_chat_template(repeated_chat_lst,
                                                 add_generation_prompt=True,
                                                 padding=True,
                                                 truncation=True,
                                                 max_length=config.rollout.prompt_length,
                                                 return_tensors='pt',
                                                 return_dict=True,
                                                 tokenize=True)
            
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            position_ids = compute_position_id_with_mask(attention_mask)

            batch_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}
            print(f'main_gen.py, input_ids.shape = {input_ids.shape}, content:', input_ids[:, -32:-26])

            data = DataProto.from_dict(batch_dict)
            real_batch_size = data.batch['input_ids'].shape[0]
            
            original_dp_size = dp_size
            dp_size = wg.world_size  # ray会先把数据分到每个worker，再每个tp group内收集，所以要保证总数能被worker数整除，该校验不应考虑tp
            if real_batch_size % dp_size != 0:
                dummy_data_size = dp_size - real_batch_size % dp_size
                dummy_data = data[:dummy_data_size]
                data = DataProto.concat([data, dummy_data])
                print(
                    f'dp_size {dp_size} is not divisible by real_batch_size {real_batch_size}, add {dummy_data_size} dummy data'
                )
            dp_size = original_dp_size

            batch_size = data.batch['input_ids'].shape[0]
            assert batch_size % dp_size == 0, f'batch_size {batch_size} is not divisible by dp_size {dp_size}'

            print(f'[{batch_idx+1}/{num_batch}] Start to generate.')
            
            # Generate all samples at once
            print('ZHS batch len:', len(data.batch['input_ids']))
            output = wg.generate_sequences(data)
            # Remove dummy data
            output = output[:real_batch_size]
            output_text = tokenizer.batch_decode(output.batch['input_ids'][:, -config.rollout.response_length:],
                                               skip_special_tokens=False)

            # Remove padding
            pad_token = tokenizer.pad_token
            output_text_unpad = []
            for text in output_text:
                output_text_unpad.append(text.replace(pad_token, ''))

            output_lst.extend(output_text_unpad)

        # Reshape output_lst from (total_samples,) to (n_data, n_samples)
        total_samples = len(output_lst)
        n_data = total_samples // config.data.n_samples
        output_lst = sum([output_lst[i::n_data] for i in range(n_data)], start=[])  # 恢复负载均衡原始顺序
        output_lst = np.array(output_lst).reshape(n_data, config.data.n_samples).tolist()

        # Add to the data frame
        dataset['responses'] = output_lst

        # add correctness field
        total_lst = compute_correctness(dataset)
        dataset['correctness'] = total_lst

        # Write to a new parquet
        output_dir = os.path.dirname(config.data.output_path)
        makedirs(output_dir, exist_ok=True)
        dataset.to_json(config.data.output_path, orient='records', force_ascii=False, lines=True)
    
    if 'correctness' not in dataset:
        total_lst = compute_correctness(dataset)
        dataset['correctness'] = total_lst
        dataset.to_json(config.data.output_path, orient='records', force_ascii=False, lines=True)
        print(f"Output file {config.data.output_path} doesn't have correctness field. Have computed each answer's correctness and saved.")
    
    output_dir = os.path.dirname(config.data.output_path)
    # Compute evaluation metrics
    prompts = dataset[config.data.prompt_key]
    responses = dataset['responses']  # Using the generated responses
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    # 统计长度均值和截断比例
    output_lst = [str(r) for responses_this in list(responses) for r in responses_this]
    # print(output_lst)
    print(type(output_lst), type(output_lst[0]))
    unpad_tokenized = tokenizer(output_lst, add_special_tokens=False).input_ids
    len_response_tokens = [len(tokens) for tokens in unpad_tokenized]
    len_mean = np.mean(len_response_tokens)
    cutoff_ratio = sum([l == config.rollout.response_length for l in len_response_tokens]) / len(unpad_tokenized)
    print('length cutoff ratio:', cutoff_ratio)

    passes = 0
    total = len(dataset)
    total_scores = []
    conses = 0
    
    for i in range(total):
        response_lst = responses[i]
        data_source = data_sources[i]
        prompt = prompts[i]
        reward_data = reward_model_data[i]
        reward_fn = select_reward_fn(data_source)
        ground_truth = reward_data['ground_truth']
        score_lst = []
        for r in response_lst:
            try:
                if config.data.skip_format_reward:
                    score = reward_fn(r, ground_truth, skip_format_reward=True)
                else:
                    score = reward_fn(r, ground_truth, skip_format_reward=False)
            except:  # 没字段表示没指定该参数，默认跳过格式校验
                score = reward_fn(r, ground_truth, skip_format_reward=True)
            score_lst.append(score)
        max_score = np.max(score_lst)
        total_scores.append(score_lst)
        if max_score == 1:
            passes += 1
        
        extracted_lst = [extract_answer(r) for r in response_lst]
        extracted_lst = [r for r in extracted_lst if r is not None]
        cons_answers = find_mode(extracted_lst)
        cons_response_lst = [r for r in response_lst if extract_answer(r) in cons_answers]
        is_cons_correct_list = list()
        for r in cons_response_lst:
            try:
                if config.data.skip_format_reward:
                    score = reward_fn(r, ground_truth, skip_format_reward=True)
                else:
                    score = reward_fn(r, ground_truth, skip_format_reward=False)
            except:  # 没字段表示没指定该参数，默认跳过格式校验
                score = reward_fn(r, ground_truth, skip_format_reward=True)
            is_cons_correct_list.append(score)
        if any(is_cons_correct_list):
            conses += np.mean(is_cons_correct_list)

    n_samples = config.data.n_samples
    pass_at_n = passes / total
    pass_at_1 = np.mean(total_scores)
    cons_at_n = conses / total

    spent_time = time.time() - start_time
    spent_hours = spent_time / 60 / 60
    # Save metrics to CSV
    csv_path = os.path.join(output_dir, f'pass_{spent_hours:.2f}h.csv')
    
    # Prepare the row data
    # Extract the dataset name from the path
    dataset_name = os.path.basename(config.data.path)
    row_data = {
        'model_path': config.model.path,
        'dataset': dataset_name,
        'pass@1': pass_at_1,
        f'pass@{n_samples}': pass_at_n,
        f'cons@{n_samples}': cons_at_n,
        'cutoff_raio': cutoff_ratio,
        'mean_response_tokens': len_mean,
        'run_hours': spent_hours
    }

    # Check if file exists
    file_exists = os.path.isfile(csv_path)
    
    # Write to CSV
    with open(csv_path, mode='a', newline='') as f:  # 追加写，不会覆盖，所以没事
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

    # Convert the row data into a list of lists format for tabulate
    table_data = [[k, v] for k, v in row_data.items()]
    
    # Print table
    print(tabulate(table_data, headers=['Metric', 'Value'], tablefmt='grid'))


def compute_correctness(dataset):
    total_lst = list()
    for i in range(len(dataset)):
        row = dataset.iloc[i]
        prompt = row['prompt']
        gt = row['reward_model']['ground_truth']
        # print(gt)
        responses_this = row['responses']

        true_false = [int(deepscaler_reward_fn(response, gt, skip_format_reward=True)) for response in responses_this]
        total_lst.append(true_false)
    return total_lst


def find_mode(lst):
    if len(lst) == 0:
        return list()
    counter = Counter(lst)
    max_count = max(counter.values())
    mode = [k for k, v in counter.items() if v == max_count]
    return mode


# Add the select_reward_fn from main_eval.py
def select_reward_fn(data_source):
    if data_source == 'lighteval/MATH':
        from verl.utils.reward_score import math
        return math.compute_score
    else:
        from deepscaler.rewards.math_reward import deepscaler_reward_fn
        return deepscaler_reward_fn

if __name__ == '__main__':
    main()
