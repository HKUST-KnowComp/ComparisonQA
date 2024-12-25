import os
import torch
import transformers
import json
import re
from tqdm import tqdm

with open('comparisonqa_benchmark/comparisonqa_test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

mode = "few"  # or "zero"

# Check device status
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('CUDA available:', torch.cuda.is_available())
print(torch.cuda.get_device_name())
print('Device number:', torch.cuda.device_count())
print(torch.cuda.get_device_properties(device))
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
    torch.cuda.set_device(1)

model_name = 'google/gemma-2-9b'
# model_name = 'tiiuae/falcon-11B'
# model_name = 'mistralai/Mistral-7B-v0.3'
# model_name = 'mistralai/Mistral-7B-Instruct-v0.3'
# model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
# model_name = 'meta-llama/Llama-3.1-8B-Instruct'
# model_name = 'meta-llama/Llama-3.2-3B-Instruct'

name = model_name.split('/')[-1].replace('.', '').replace('-', '_')
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def generate_text(prompt):
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
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


zero_prompt = """{} A. {}.  B. {}.  C. {}.  D. {}
The correct answer is: """

few_prompt = """Answer the following multiple choice question. Select only one correct answer from the choices.
Follow these examples:

Which country does Cimarron Firearms belong to? A. Australia.  B. Mexico.  C. Canada.  D. America.
Answer: **D.**

Which specific target does Belimumab inhibit or act against? A. B-cell activating factor (BAFF).  B. Human Rhesus factor.  C. Tumor Necrosis Factor (TNF).  D. Programmed death-1 (PD-1).
Answer: **A.**

In which geographical regions can Leuconotopicus typically be found? A. Central and South America.  B. North and South America.  C. Only North America.  D. Europe and Asia.
Answer: **B.**

What is the primary color range associated with Ochre? A. Blue to green.  B. Yellow-brown to bright red.  C. Yellow to deep orange or brown.  D. Purple to black.
Answer: **C.**

{} A. {}.  B. {}.  C. {}.  D. {}.
"""

new_data = []
print_count = 0
for line in tqdm(data):
    id = line["question_id"]
    for q in ["high_question", "low_question"]:
        question = line[q]["question"]
        if mode == "few":
            input = few_prompt.format(question, line[q]["options"]["A"], line[q]["options"]["B"], line[q]["options"]["C"], line[q]["options"]["D"])
        else:
            input = zero_prompt.format(question, line[q]["options"]["A"], line[q]["options"]["B"], line[q]["options"]["C"], line[q]["options"]["D"])
        generated = generate_text(input)

        input_length = len(input)
        output_start_index = generated.find(input) + input_length
        answer = generated[output_start_index:].strip()

        line[q]["model_output"] = answer
    new_data.append(line)

    print_count += 1
    if print_count % 1000 == 1:
        with open(f'./experiments/longtailqa_test_output_{name}_{mode}.json', 'w', encoding='utf-8') as file:
            json.dump(new_data, file, ensure_ascii=False, indent=4)
        with open(f'./experiments/longtailqa_test_output_{name}_{mode}_copy.json', 'w', encoding='utf-8') as file:
            json.dump(new_data, file, ensure_ascii=False, indent=4)

with open(f'./experiments/longtailqa_test_output_{name}_{mode}.json', 'w', encoding='utf-8') as file:
    json.dump(new_data, file, ensure_ascii=False, indent=4)
with open(f'./experiments/longtailqa_test_output_{name}_{mode}_copy.json', 'w', encoding='utf-8') as file:
    json.dump(new_data, file, ensure_ascii=False, indent=4)
