import glob
import json

from openai import OpenAI
from tqdm import tqdm
import sys

sys.path.append('../../../')
from LLM_api.key import openai_key

import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()

client = OpenAI(api_key=openai_key)

output_list = []

response = client.batches.list()
for r in tqdm(response):
    if not r.metadata:
        continue
    if r.metadata['description'].startswith("Comparisonqa Answer Test 4omini Fewshot Batch") and r.status == "completed":
        batch_number = r.metadata['description'].split()[-1]
        output_file_id = r.output_file_id
        output_list.append((batch_number, output_file_id))
print("Total number of completed comparisonqa answer batches: {}".format(len(output_list)))
print(output_list)

for batch_number, output_file_id in output_list:
    file_response = client.files.content(output_file_id)
    with open("comparisonqa_answer_test_4omini_fewshot_batch_{}_output_file.jsonl".format(batch_number), 'wb') as f:
        f.write(file_response.content)
        print("Batch {} output file saved".format(batch_number))

generated_comparisonqas = glob.glob("comparisonqa_answer_test_4omini_fewshot_batch_*_output_file.jsonl")

total_comparisonqas = {}

for f in tqdm(generated_comparisonqas, desc="Parsing generated comparisonqa"):
    if 'SAMPLE' in f:
        continue
    data = [json.loads(line) for line in open(f, 'r').readlines()]
    for d in data:
        question_id = d['custom_id']
        generations = d['response']['body']['choices'][0]['message']['content']
        if question_id not in total_comparisonqas:
            total_comparisonqas[question_id] = generations
        else:
            total_comparisonqas[question_id] += generations

with open("comparisonqa_answer_test_4omini_fewshot_total_output.json", 'w') as f:
    json.dump(total_comparisonqas, f, indent=4)
    print(
        "Total number of {} comparisonqa answer saved".format(len(total_comparisonqas)))
