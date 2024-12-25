import json
import re

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="keyword")
args = parser.parse_args()

name = args.model_name.split('/')[-1].replace('.', '').replace('-', '_')
print(name)

with open(f'../experiments/longtailqa_test_perplexity_{name}_zero.json', 'r', encoding='utf-8') as f:
    perplexity_data = json.load(f)


label_dic = {"A": 0, "B": 1, "C": 2, "D": 3}
perplexity_avg = 0
for line in perplexity_data:
    high_gold = list(line["high_question"]["answer"].keys())[0]
    low_gold = list(line["low_question"]["answer"].keys())[0]
    high_p = line["high_question"]["perplexity_gen"][label_dic[high_gold]]
    low_p = line["low_question"]["perplexity_gen"][label_dic[low_gold]]
    perplexity_avg += high_p
    perplexity_avg += low_p
perplexity_avg = perplexity_avg / (2*len(perplexity_data))

print(perplexity_avg)

with open(f'../experiments/longtailqa_rk_output_{name}.json', 'r', encoding='utf-8') as f:
    predict_all = json.load(f)


output_name = "model_output"
print(predict_all[-1])
# assert len(predict_all) == len(gold_all)*2
num = 0
high_correct = 0
low_correct = 0
label_dic = {"A": 0, "B": 1, "C": 2, "D": 3, None: 4}
label_dic_reverse = {0: "A", 1: "B", 2: "C", 3: "D"}

again_count = {"high": 0, "low": 0}
fail_count_2r = {"high": 0, "low": 0}

def max_overlapping_count(options):
    keys = list(options.keys())
    max_count = 0
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            count = len(set(re.split(r'[ .]+', options[keys[i]])) & set(re.split(r'[ .]+', options[keys[j]])))
            max_count = max(max_count, count)
    return max_count

def find_unique_exceeding_count(options, output):
    options = {key: value.lower() for key, value in options.items()}
    output = output.lower()
    max_count = max_overlapping_count(options)
    exceeding_keys = []
    for key in options.keys():
        count = len(set(re.split(r'[ .]+', options[key])) & set(re.split(r'[ .]+', output)))
        if count > max_count:
            exceeding_keys.append(key)
    return exceeding_keys[0] if len(exceeding_keys) == 1 else None

def extract_prediction(output, options):
    matches = re.findall(r'<strong>(.*?)</strong>', output)
    if matches:
        prediction = matches[0].split(".")[0]
        if prediction in ["A", "B", "C", "D"]:
            return prediction

    matches = re.findall(r'\*\*(.*?).\*\*', output)
    if matches:
        prediction = matches[0].split(".")[0]
        if prediction in ["A", "B", "C", "D"]:
            return prediction

    matches = re.findall(r'Answer:\s*([A-D])', output)
    if matches:
        prediction = matches[0].strip()
        if prediction in ["A", "B", "C", "D"]:
            return prediction

    prediction = output.split("\n")[0].split(".")[0]
    if prediction in ["A", "B", "C", "D"]:
        return prediction

    prediction = output.split("\n")[0].strip(".")
    for id, option in options.items():
        if option.startswith(prediction) or option.endswith(prediction):
            prediction = id
            break
    if prediction in ["A", "B", "C", "D"]:
        return prediction

    prediction = output.split(".")[0]
    for id, option in options.items():
        if option.startswith(prediction):
            prediction = id
            break
    if prediction in ["A", "B", "C", "D"]:
        return prediction

    prediction = output.split("\n")[0].split(" - ")[0]
    if prediction in ["A", "B", "C", "D"]:
        return prediction

    matches = re.findall(r'\b([ABCD])\b[^\w]*', output)
    if matches:
        unique_letters = set(matches)
        if len(unique_letters) == 1:
            prediction = unique_letters.pop()
    if prediction in ["A", "B", "C", "D"]:
        return prediction

    matches = re.findall(r'\b([ABCD])\b[^\w]*', output.split("\n")[0])
    if matches:
        unique_letters = set(matches)
        if len(unique_letters) == 1:
            prediction = unique_letters.pop()
    if prediction in ["A", "B", "C", "D"]:
        return prediction

    prediction = find_unique_exceeding_count(options, output)
    if prediction in ["A", "B", "C", "D"]:
        return prediction

    return None


def extract_2r_prediction(q_type, gold, question_line, ppl):
    global again_count, fail_count_2r
    predict = extract_prediction(question_line[output_name], question_line["options"])
    if (ppl >= perplexity_avg and predict == gold) or (ppl <= perplexity_avg and predict != gold):
        again_count[q_type] += 1
        predict = gold
        for o, a in question_line["statement_answer"].items():
            if o == gold:
                if "true" not in a.lower():
                    predict = None
            else:
                if "true" in a.lower():
                    predict = None
        ans = list(question_line["statement_answer"].values())
        if not any("true" in s.lower() or "false" in s.lower() for s in ans):
            fail_count_2r[q_type] += 1
    return predict

fail_count_high_orig = 0
fail_count_low_orig = 0
num_orig = 0
high_correct_orig = 0
low_correct_orig = 0
for line in predict_all:
    num_orig += 1
    high_predict = extract_prediction(line["high_question"][output_name], line["high_question"]["options"])
    low_predict = extract_prediction(line["low_question"][output_name], line["high_question"]["options"])
    high_gold = list(line["high_question"]["answer"].keys())[0]
    low_gold = list(line["low_question"]["answer"].keys())[0]
    if high_predict is None:
        print(line["high_question"])
        print(line["high_question"]["options"][high_gold])
        print(line["high_question"][output_name])
        print("\n")
        fail_count_high_orig += 1
    if low_predict is None:
        print(line["low_question"])
        print(line["low_question"]["options"][low_gold])
        print(line["low_question"][output_name])
        print("\n")
        fail_count_low_orig += 1
    if high_predict == high_gold:
        high_correct_orig += 1
    if low_predict == low_gold:
        low_correct_orig += 1

print(fail_count_high_orig)
print(fail_count_low_orig)
print(num_orig)
high_acc_orig = high_correct_orig*100 / num_orig
low_acc_orig = low_correct_orig*100 / num_orig
print("orig high ACC Score:", high_acc_orig)
print("orig low ACC Score:", low_acc_orig)
print("orig difference ACC Score:", high_acc_orig - low_acc_orig)


for i in range(len(predict_all)):
    line = predict_all[i]
    ppl_line = perplexity_data[i]
    assert line["question_id"] == ppl_line["question_id"]
    num += 1
    high_gold = list(line["high_question"]["answer"].keys())[0]
    low_gold = list(line["low_question"]["answer"].keys())[0]
    high_p = ppl_line["high_question"]["perplexity_gen"][label_dic[high_gold]]
    low_p = ppl_line["low_question"]["perplexity_gen"][label_dic[low_gold]]

    high_predict = extract_2r_prediction("high", high_gold, line["high_question"], high_p)
    if high_predict == high_gold:
        high_correct += 1

    low_predict = extract_2r_prediction("low", low_gold, line["low_question"], low_p)
    if low_predict == low_gold:
        low_correct += 1


print(f"again_count: {again_count}")
again_count['high'] /= num
again_count['low'] /= num
print(f"again_count ratio: {again_count}")
print(f"fail_count_2r: {fail_count_2r}")
print(num)
high_acc = high_correct*100 / num
low_acc = low_correct*100 / num
print("high ACC Score:", high_acc)
print("low ACC Score:", low_acc)
print("difference ACC Score:", high_acc - low_acc)

print("diff high ACC Score:", high_acc-high_acc_orig)
print("diff low ACC Score:", low_acc-low_acc_orig)
print("diff difference ACC Score:", (high_acc - low_acc)-(high_acc_orig - low_acc_orig))



