import os.path
import time

from openai import OpenAI
import sys
import argparse

sys.path.append('../../../')
from LLM_api.key import openai_key

parser = argparse.ArgumentParser()
parser.add_argument('--batch_start_id', type=int, default=1)
parser.add_argument('--batch_end_id', type=int, default=1)
# parser.add_argument('--model_type', type=int, default=0)   # 0 4o-mini    1 4o
args = parser.parse_args()

assert args.batch_start_id <= args.batch_end_id

client = OpenAI(api_key=openai_key)

for id in range(args.batch_start_id, args.batch_end_id + 1):
    if os.path.exists("comparisonqa_answer_test_4omini_fewshot_batch_{}_input_file_id.txt".format(id)):
        with open("comparisonqa_answer_test_4omini_fewshot_batch_{}_input_file_id.txt".format(id), 'r') as f:
            batch_input_file_id = f.read().strip()
            print("Batch {} already exists with input file id {}".format(id, batch_input_file_id))
            continue

    batch_input_file = client.files.create(
        file=open("comparisonqa_answer_test_4omini_fewshot_batch_{}.jsonl".format(id), "rb"),
        purpose="batch"
    )

    batch_input_file_id = batch_input_file.id

    response = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "Comparisonqa Answer Test 4omini Fewshot Batch {}".format(id)
        }
    )

    batch_id = response.id
    print("Batch {} created with batch id {}".format(id, batch_id))

    while client.batches.retrieve(batch_id).status not in ["completed", "finalizing"]:
        print("Batch {} is still running at time {}".format(id, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        time.sleep(600)

    output_response = client.batches.retrieve(batch_id)

    print("Batch {} is completed".format(id))

    with open("comparisonqa_answer_test_4omini_fewshot_batch_{}_response_info.txt".format(id), 'w') as f:
        f.write(str(response))
        f.write('\n\n')
        f.write(str(output_response))
        print("Batch {} response info saved".format(id))

print("All batches are completed for generation")
