import os
import torch
import transformers
import json
import re
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="keyword")
args = parser.parse_args()

name = args.model_name.split('/')[-1].replace('.', '').replace('-', '_')
print(name)

with open('comparisonqa_benchmark/comparisonqa_test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open('comparisonqa_benchmark/comparisonqa_test_option_statement.json', 'r', encoding='utf-8') as f:
    statement_data = json.load(f)

label_dic = {"A": 0, "B": 1, "C": 2, "D": 3}
# Check device status
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('CUDA available:', torch.cuda.is_available())
# print(torch.cuda.get_device_name())
# print('Device number:', torch.cuda.device_count())
# print(torch.cuda.get_device_properties(device))
# if torch.cuda.is_available():
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(2)
#     torch.cuda.set_device(2)


tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")
model.eval()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)


def generate_text(prompt):
    inputs = tokenizer.encode(prompt, return_tensors='pt').to("cuda")
    length = len(inputs[0])
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=length+20,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

prompt = """Judge whether the statement is True or False, keep the answer in one word:
Statement: The specific target Belimumab inhibits or acts against is B-cell activating factor (BAFF).  Answer: True
Statement: The country Cimarron Firearms belongs to is Canada.  Answer: False
Statement: The geographical regions where Leuconotopicus can typically be found are North and South Africa.  Answer: False
Statement: The primary color range associated with Ochre is Yellow to deep orange or brown.  Answer: True
Statement: {}  Answer: """

new_data = []
print_count = 0
statement_count = {"high_question": 0, "low_question": 0}
total_len = len(data)
for line_id in tqdm(range(total_len)):
    line = data[line_id]
    question_id = line["question_id"]
    for q in ["high_question", "low_question"]:
        gold = list(line[q]["answer"].keys())[0]
        statement_count[q] += 1
        answer = {}
        for option in label_dic:
            option_id = question_id+"_"+q.split("_")[0]+"_"+option
            statement = statement_data[option_id]
            input = prompt.format(statement)
            generated = generate_text(input)
            input_length = len(input)
            output_start_index = generated.find(input) + input_length
            answer[option] = generated[output_start_index:].strip()
        line[q]["statement_answer"] = answer
    new_data.append(line)

    print_count += 1
    if print_count % 1000 == 1:
        with open(f'./experiments/longtailqa_rk_output_{name}.json', 'w', encoding='utf-8') as file:
            json.dump(new_data, file, ensure_ascii=False, indent=4)
        with open(f'./experiments/longtailqa_rk_output_{name}_copy.json', 'w', encoding='utf-8') as file:
            json.dump(new_data, file, ensure_ascii=False, indent=4)

with open(f'./experiments/longtailqa_rk_output_{name}.json', 'w', encoding='utf-8') as file:
    json.dump(new_data, file, ensure_ascii=False, indent=4)
with open(f'./experiments/longtailqa_rk_output_{name}_copy.json', 'w', encoding='utf-8') as file:
    json.dump(new_data, file, ensure_ascii=False, indent=4)

print(statement_count)
