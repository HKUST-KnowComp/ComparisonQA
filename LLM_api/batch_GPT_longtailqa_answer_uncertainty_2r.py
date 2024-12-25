import json
import re
from tqdm import tqdm

with open('./longtailqa_answer_test_uncertainty_4omini_fewshot_total_output.json', 'r', encoding='utf-8') as f:
    predict_all = json.load(f)
    print(len(predict_all))

with open('../comparisonqa_benchmark/comparisonqa_test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open('../comparisonqa_benchmark/comparisonqa_test_option_statement.json', 'r', encoding='utf-8') as f:
    statement_data = json.load(f)

high_ppl_list = []
low_ppl_list = []
fail_count_high = 0
fail_count_low = 0
ppl_dict = {}
for line in data:
    try:
        high_ppl = int(re.findall(r'(\d+)%', predict_all[line["question_id"]+"_high"])[0])
        high_ppl_list.append(high_ppl)
    except:
        # print(predict_all[line["question_id"]+"_high"])
        fail_count_high += 1
        high_ppl = None
    try:
        low_ppl = int(re.findall(r'(\d+)%', predict_all[line["question_id"] + "_low"])[0])
        low_ppl_list.append(low_ppl)
    except:
        # print(predict_all[line["question_id"]+"_low"])
        fail_count_low += 1
        low_ppl = None
    ppl_dict[line["question_id"]+"_high"] = high_ppl
    ppl_dict[line["question_id"]+"_low"] = low_ppl

print(fail_count_high)
print(fail_count_low)
ppl_avg = (sum(high_ppl_list)+sum(low_ppl_list)) / (len(high_ppl_list)+len(low_ppl_list))
print(ppl_avg)

label_dic = {"A": 0, "B": 1, "C": 2, "D": 3, None: 4}

fail_count_high = 0
fail_count_low = 0
again_count_2r = {"high": 0, "low": 0}


prompt_2r = """Judge whether the statement is True or False, keep the answer in one word:
Statement: The specific target Belimumab inhibits or acts against is B-cell activating factor (BAFF).  Answer: True
Statement: The country Cimarron Firearms belongs to is Canada.  Answer: False
Statement: The geographical regions where Leuconotopicus can typically be found are North and South Africa.  Answer: False
Statement: The primary color range associated with Ochre is Yellow to deep orange or brown.  Answer: True
Statement: {}  Answer: """

for t in tqdm(range(len(data)), desc="Fusing prompts with entities"):
    for k in ["high", "low"]:
        statement_qa = {}
        for o in ["A", "B", "C", "D"]:
            statement_qa[o] = prompt_2r.format(statement_data[f"{data[t]['question_id']}_{k}_{o}"])
        data[t][f"{k}_question"]["statement_qa"] = statement_qa

total_batch_requests = []
for line in data:
    pattern = r'\*\*(.*?).\*\*'
    high_gold = list(line["high_question"]["answer"].keys())[0]
    low_gold = list(line["low_question"]["answer"].keys())[0]
    try:
        high_ppl = int(re.findall(r'(\d+)%', predict_all[line["question_id"] + "_high"])[0])
        try:
            high_predict = re.findall(r'([A-D])\.', predict_all[line["question_id"] + "_high"])[0]
            assert high_predict in ["A", "B", "C", "D"]
        except:
            try:
                high_predict = re.findall(r'([A-D])\]\.', predict_all[line["question_id"] + "_high"])[0]
                assert high_predict in ["A", "B", "C", "D"]
            except:
                print(predict_all[line["question_id"] + "_high"])
                high_predict = None
        if (high_ppl <= ppl_avg and high_predict != high_gold) or (high_ppl >= ppl_avg and high_predict == high_gold):
            again_count_2r["high"] += 1

            for o in ["A", "B", "C", "D"]:
                total_batch_requests.append(
                    {
                        "custom_id": f"{line['question_id']}_high_{o}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": "gpt-4o-mini",
                            "messages": [
                                {"role": "user", "content": line["high_question"]['statement_qa'][o]},
                            ],
                            "max_tokens": 200,
                            "temperature": 0.8
                        }
                    }
                )
    except:
        # print(predict_all[line["question_id"]+"_high"])
        fail_count_high += 1
    try:
        low_ppl = int(re.findall(r'(\d+)%', predict_all[line["question_id"] + "_low"])[0])
        try:
            low_predict = re.findall(r'([A-D])\.', predict_all[line["question_id"] + "_low"])[0]
            assert low_predict in ["A", "B", "C", "D"]
        except:
            try:
                low_predict = re.findall(r'([A-D])\]\.', predict_all[line["question_id"] + "_low"])[0]
                assert low_predict in ["A", "B", "C", "D"]
            except:
                print(predict_all[line["question_id"] + "_low"])
                low_predict = None
        if (low_ppl <= ppl_avg and low_predict != low_gold) or (low_ppl >= ppl_avg and low_predict == low_gold):
            again_count_2r["low"] += 1

            for o in ["A", "B", "C", "D"]:
                total_batch_requests.append(
                    {
                        "custom_id": f"{line['question_id']}_low_{o}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": "gpt-4o-mini",
                            "messages": [
                                {"role": "user", "content": line["low_question"]['statement_qa'][o]},
                            ],
                            "max_tokens": 200,
                            "temperature": 0.8
                        }
                    }
                )
    except:
        # print(predict_all[line["question_id"]+"_low"])
        fail_count_low += 1

print(again_count_2r)
print("Total number of questions: ", len(total_batch_requests))

# for every 50,000 requests, save to a new jsonl file and number the file as intention_generation_batch_1.jsonl, intention_generation_batch_2.jsonl, etc.
batch_size = 50000
total_batches = len(total_batch_requests) // batch_size + 1

for i in range(total_batches):
    with open("./longtailqa_answer_test_uncertainty_2r_4omini_fewshot_left_batch_{}.jsonl".format(i + 1), 'w') as f:
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

