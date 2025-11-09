# def extract_answer(output):
#     all_substrs = [
#         "Answer",
#         "ANSWER",
#         "The",
#         "Task",
#         "Finished"
#     ]
#     for sub_str in all_substrs:
#         output = output.replace(sub_str, sub_str.lower())
    
#     extracted_r = output.split("# answer")[-1].lstrip()
#     extracted_r = extracted_r.split("#answer")[-1].lstrip()
#     # extracted_r = extracted_r.split("the answer is")[-1].strip()
#     # extracted_r = extracted_r.split("task finished.")[-1].strip()
#     extracted_r = extracted_r.split("answer:")[-1].lstrip()
#     # extracted_r = extracted_r.split("the answer")[-1].strip()

#     if len(extracted_r) > 30:
#     #     print(extracted_r)
#         extracted_r = "None"
    
#     return extracted_r
import random
import re
from datasets import load_dataset
def try_extract(output, pattern):
    matches = re.findall(pattern, output, re.DOTALL)
    answers = [match.strip() for match in matches]
    if len(answers) > 0:
        return answers[-1]
    else:
        return "None"

def extract_answer(output, model="llama"):

    answers = []
    for piece in output.split('boxed{')[1:]:
        n = 0
        for i in range(len(piece)):
            if piece[i] == '{':
                n += 1
            elif piece[i] == '}':
                n -= 1
                if n < 0:
                    if i + 1 < len(piece) and piece[i + 1] == '%':
                        answers.append(piece[: i + 1])
                    else:
                        answers.append(piece[:i])
                    break
    if len(answers) > 0:
        return answers[0]
    else:
        return "None"
        # # print("here")
        # # match = re.search(r"\\boxed\{(\d+)\}", output)
        # match = re.search(r'\\boxed{([^}]*)}', output)
        # print(match)
        # if match:
        #     answer = match.group(1)
        #     return answer

    # pattern = r"####*(.*?)\n"
    # answer = try_extract(output, pattern)
    # if answer != "None":
    #     return answer

    # pattern = r"# answer\s*(.*?)\n"
    # answer = try_extract(output, pattern)
    # if answer != "None":
    #     return answer
    
    # pattern = r"answer is:\s*(.*?)\n"
    # answer = try_extract(output, pattern)
    # if answer != "None":
    #     return answer
    
    # pattern = r"answer is\s*(.*?)\n"
    # answer = try_extract(output, pattern)
    # if answer != "None":
    #     return answer


    
    # # else:
    # #     return ""

    return "None"


def extract_all_answers(output):
    all_substrs = [
        "Answer",
        "ANSWER",
        "The",
        "Task",
        "Finished"
    ]
    for sub_str in all_substrs:
        output = output.replace(sub_str, sub_str.lower())
    # print(output)
    if output[-1] != "\n":
        output += "\n"
    # 匹配所有以 '# Answer' 开头，答案位于下一行之前的内容
    pattern = r"# answer\s*(.*?)\n"
    matches = re.findall(pattern, output, re.DOTALL)
    # 返回所有匹配的答案
    answers = [match.strip() for match in matches]
    
    return answers



def is_a_positive_critique(critique):
    critique = critique.lower()
    # if critique.find("incorrect") != -1:
    #     return False
    # if critique.find("failed") != -1:
    #     return False
    # if critique.find("fails") != -1:
    #     return False
    if critique.find("reference") != -1:
        return -1
    if critique.find("judgement:") == -1:
        return -1
    critique = critique.split("judgement:")[-1]
    critique = critique.split("\n")[0]
    

    if critique.find("incorrect") != -1:
        return 0
    if critique.find("correct") != -1:
        return 1
    
    return -1


def random_truncate(text, sep="\n\n"):

    steps = text.split(sep)
    # num = len(steps) - random.choice([1, 2])#
    num = int(random.uniform(0.2, 0.8) * len(steps))

    return sep.join(steps[0:num])

def get_example(index):
    ds = load_dataset("GAIR/o1-journey")['train']
    item = ds[index]
    question = item['question']
    text = item['longCOT']

    solution = text.split("####")[0].strip().strip("\n").strip()
    answer = text.split("####")[-1].strip().strip("\n").strip()
    return question, solution, answer