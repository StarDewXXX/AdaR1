from datasets import load_from_disk

data_path = "../LLaMA-Factory/data/my_dataset/Marco-iter0-MATH-train-K-16-shortest"

data = load_from_disk(data_path)['train']

for item in data:
    # print(item['messages'][0]['content'])
    print(item['messages'][1]['content'])
    print(item['messages'][2]['content'])
    print("-"*100)
    input("continue>")