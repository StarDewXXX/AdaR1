import json
def load_eval_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data