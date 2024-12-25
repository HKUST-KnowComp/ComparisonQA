import os
import torch

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
parser.add_argument('--mode', type=str, default='few')  # or "zero"
args = parser.parse_args()

name = args.model_name.split('/')[-1].replace('.', '').replace('-', '_')
# Check device status
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('CUDA available:', torch.cuda.is_available())
print(torch.cuda.get_device_name())
print('Device number:', torch.cuda.device_count())
print(torch.cuda.get_device_properties(device))
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.cuda.set_device(0)

import lmppl
import json
import re
from tqdm import tqdm

with open('comparisonqa_benchmark/comparisonqa_test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

scorer = lmppl.LM(args.model, device_map="cuda:0", low_cpu_mem_usage=True)

few_prompt = """Which country does Cimarron Firearms belong to? Answer: America
Which specific target does Belimumab inhibit or act against? Answer: B-cell activating factor (BAFF)
In which geographical regions can Leuconotopicus typically be found? Answer: North and South America
What is the primary color range associated with Ochre? Answer: Yellow to deep orange or brown

Answer the question following these examples:
Question: {}
Answer: {}"""

zero_prompt = """Question: {}  Answer: {}"""

new_data = []
print_count = 0
for line in tqdm(data):
    id = line["question_id"]
    for q in ["high_question", "low_question"]:
        question = line[q]["question"]
        score = []
        for o in ["A", "B", "C", "D"]:
            answer = line[q]["options"][o]
            if args.mode == "few":
                input_text = few_prompt.format(question, answer)
            else:
                input_text = zero_prompt.format(question, answer)
            ppl = scorer.get_perplexity(input_text)
            score.append(ppl)
        line[q]["perplexity_gen"] = score
    new_data.append(line)

    print_count += 1
    if print_count % 1000 == 1:
        with open(f'./experiments/comparisonqa_test_perplexity_{name}_{args.mode}.json', 'w', encoding='utf-8') as file:
            json.dump(new_data, file, ensure_ascii=False, indent=4)
        with open(f'./experiments/comparisonqa_test_perplexity_{name}_{args.mode}_copy.json', 'w', encoding='utf-8') as file:
            json.dump(new_data, file, ensure_ascii=False, indent=4)

with open(f'./experiments/comparisonqa_test_perplexity_{name}_{args.mode}.json', 'w', encoding='utf-8') as file:
    json.dump(new_data, file, ensure_ascii=False, indent=4)
with open(f'./experiments/comparisonqa_test_perplexity_{name}_{args.mode}_copy.json', 'w', encoding='utf-8') as file:
    json.dump(new_data, file, ensure_ascii=False, indent=4)
