import glob
import json
import re
from tqdm import tqdm

with open('../comparisonqa_benchmark/comparisonqa_test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

few_prompt = """Question: Which country does Cimarron Firearms belong to? A. Australia.  B. Mexico.  C. Canada.  D. America.
Answer: **D.**
Uncertainty: **30%**

Question: Which specific target does Belimumab inhibit or act against? A. B-cell activating factor (BAFF).  B. Human Rhesus factor.  C. Tumor Necrosis Factor (TNF).  D. Programmed death-1 (PD-1).
Answer: **A.**
Uncertainty: **80%**

Question: In which geographical regions can Leuconotopicus typically be found? A. Central and South America.  B. North and South America.  C. Only North America.  D. Europe and Asia.
Answer: **B.**
Uncertainty: **70%**

Question: What is the primary color range associated with Ochre? A. Blue to green.  B. Yellow-brown to bright red.  C. Yellow to deep orange or brown.  D. Purple to black.
Answer: **C.**
Uncertainty: **20%**


Answer the following multiple choice question. Select only one correct answer from the choices and give your uncertainty score. The above examples only show the format and do not really represent the true answer and uncertainty.
Question: {} A. {}.  B. {}.  C. {}.  D. {}.
Answer: """

zero_prompt = """Question: [question]
Answer: **[option].**
Uncertainty: **[uncertainty percentage]**

Answer the following multiple choice question. Select only one correct answer from the choices and give your uncertainty score, following the above format.
Question: {} A. {}.  B. {}.  C. {}.  D. {}
Answer: """

cot_prompt = """Question: [question]
Rational: [rationale]
Answer: **[option].**
Uncertainty: **[uncertainty percentage]**

Answer the following multiple choice question. Think step by step and generate a short rationale to support your reasoning. Choose one best answer based on the generated rational and give your uncertainty score, following the above format. Keep your whole response in 50 tokens.
Question: {} A. {}.  B. {}.  C. {}.  D. {}
"""


for t in tqdm(range(len(data)), desc="Fusing prompts with entities"):
    data[t]['high_prompt'] = cot_prompt.format(data[t]["high_question"]["question"], data[t]["high_question"]["options"]["A"], data[t]["high_question"]["options"]["B"], data[t]["high_question"]["options"]["C"], data[t]["high_question"]["options"]["D"])
    data[t]['low_prompt'] = cot_prompt.format(data[t]["low_question"]["question"], data[t]["low_question"]["options"]["A"], data[t]["low_question"]["options"]["B"], data[t]["low_question"]["options"]["C"], data[t]["low_question"]["options"]["D"])
total_batch_requests = []

for t in tqdm(range(len(data)), desc="Generating batch requests"):
    total_batch_requests.append(
        {
            "custom_id": data[t]['question_id']+"_high",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": data[t]['high_prompt']},
                ],
                "max_tokens": 200,
                "temperature": 0.8
            }
        }
    )
    total_batch_requests.append(
        {
            "custom_id": data[t]['question_id']+"_low",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": data[t]['low_prompt']},
                ],
                "max_tokens": 200,
                "temperature": 0.8
            }
        }
    )


print("Total number of questions: ", len(total_batch_requests))

# for every 50,000 requests, save to a new jsonl file and number the file as intention_generation_batch_1.jsonl, intention_generation_batch_2.jsonl, etc.
batch_size = 10000
total_batches = len(total_batch_requests) // batch_size + 1

for i in range(total_batches):
    with open("./comparisonqa_answer_test_uncertainty_4omini_cot_batch_{}.jsonl".format(i + 1), 'w') as f:
        for j in total_batch_requests[i * batch_size:(i + 1) * batch_size]:
            f.write(json.dumps(j) + '\n')


import tiktoken
from tqdm import tqdm
encoding = tiktoken.encoding_for_model("gpt-4o-mini")
total_len = 0
for line in tqdm(total_batch_requests):
    token_count = encoding.encode(line["body"]["messages"][0]["content"])
    total_len += len(token_count)
print(total_len)
