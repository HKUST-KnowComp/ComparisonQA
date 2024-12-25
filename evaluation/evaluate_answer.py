import json
import re
from collections import Counter
from tqdm import tqdm
import random
import numpy as np
from sklearn.metrics import f1_score

with open('../experiments/longtailqa_test_output_Meta_Llama_3_8B_Instruct_few.json', 'r', encoding='utf-8') as f:
    predict_all = json.load(f)

output_name = "model_output"
print(predict_all[-1])
# assert len(predict_all) == len(gold_all)*2
num = 0
high_correct = 0
low_correct = 0
high_predict_label = []
low_predict_label = []
high_gold_label = []
low_gold_label = []
label_dic = {"A": 0, "B": 1, "C": 2, "D": 3, None: 4}

fail_count_high = 0
fail_count_low = 0

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


for line in predict_all:
    num += 1
    high_predict = extract_prediction(line["high_question"][output_name], line["high_question"]["options"])
    low_predict = extract_prediction(line["low_question"][output_name], line["high_question"]["options"])
    high_gold = list(line["high_question"]["answer"].keys())[0]
    low_gold = list(line["low_question"]["answer"].keys())[0]
    if high_predict is None:
        print(line["high_question"])
        print(line["high_question"]["options"][high_gold])
        print(line["high_question"][output_name])
        print("\n")
        fail_count_high += 1
    if low_predict is None:
        print(line["low_question"])
        print(line["low_question"]["options"][low_gold])
        print(line["low_question"][output_name])
        print("\n")
        fail_count_low += 1
    high_gold_label.append(label_dic[high_gold])
    low_gold_label.append(label_dic[low_gold])
    high_predict_label.append(label_dic[high_predict])
    low_predict_label.append(label_dic[low_predict])
    if high_predict == high_gold:
        high_correct += 1
    if low_predict == low_gold:
        low_correct += 1

print(fail_count_high)
print(fail_count_low)
print(num)
high_acc = high_correct / num
low_acc = low_correct / num
print("high ACC Score:", high_acc)
print("low ACC Score:", low_acc)
print("difference ACC Score:", high_acc - low_acc)

high_gold_label = np.array(high_gold_label)
low_gold_label = np.array(low_gold_label)
high_predict_label = np.array(high_predict_label)
low_predict_label = np.array(low_predict_label)

high_ma_f1 = f1_score(high_gold_label, high_predict_label, labels=[0,1,2,3], average='macro')
low_ma_f1 = f1_score(low_gold_label, low_predict_label, labels=[0,1,2,3], average='macro')
print("high Macro F1 Score:", high_ma_f1)
print("low Macro F1 Score:", low_ma_f1)
print("avg Macro F1 Score:", (high_ma_f1 + low_ma_f1)/2)
print("difference Macro F1 Score:", high_ma_f1 - low_ma_f1)

