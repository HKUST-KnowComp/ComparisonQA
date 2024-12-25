import json
import re
from tqdm import tqdm
import random
import numpy as np
from sklearn.metrics import f1_score

with open('../LLM_api/comparisonqa_answer_test_uncertainty_4omini_few_total_output.json', 'r', encoding='utf-8') as f:
    predict_all = json.load(f)

with open('../comparisonqa_benchmark/comparisonqa_test.json', 'r', encoding='utf-8') as f:
    gold_all = json.load(f)

num = 0
high_correct = 0
low_correct = 0
high_ppl_list = []
low_ppl_list = []
high_predict_label = []
low_predict_label = []
high_gold_label = []
low_gold_label = []
label_dic = {"A": 0, "B": 1, "C": 2, "D": 3, None: 4}

fail_count_high = 0
fail_count_low = 0

for line in gold_all:
    pattern = r'\*\*(.*?).\*\*'
    num += 1
    try:
        high_predict = re.findall(r'([A-D])\.', predict_all[line["question_id"]+"_high"])[0]
        high_ppl = int(re.findall(r'(\d+)%', predict_all[line["question_id"]+"_high"])[0])
        high_ppl_list.append(high_ppl)
        assert high_predict in ["A", "B", "C", "D"]
    except:
        try:
            high_predict = re.findall(r'([A-D])\]\.', predict_all[line["question_id"] + "_high"])[0]
            high_ppl = int(re.findall(r'(\d+)%\]', predict_all[line["question_id"] + "_high"])[0])
            high_ppl_list.append(high_ppl)
            assert high_predict in ["A", "B", "C", "D"]
        except:
            print(predict_all[line["question_id"]+"_high"])
            high_predict = None
            fail_count_high += 1
    try:
        low_predict = re.findall(r'([A-D])\.', predict_all[line["question_id"] + "_low"])[0]
        low_ppl = int(re.findall(r'(\d+)%', predict_all[line["question_id"] + "_low"])[0])
        low_ppl_list.append(low_ppl)
        assert low_predict in ["A", "B", "C", "D"]
    except:
        try:
            low_predict = re.findall(r'([A-D])\]\.', predict_all[line["question_id"] + "_low"])[0]
            low_ppl = int(re.findall(r'(\d+)%\]', predict_all[line["question_id"] + "_low"])[0])
            low_ppl_list.append(low_ppl)
            assert low_predict in ["A", "B", "C", "D"]
        except:
            print(predict_all[line["question_id"]+"_low"])
            low_predict = None
            fail_count_low += 1
    high_gold = list(line["high_question"]["answer"].keys())[0]
    low_gold = list(line["low_question"]["answer"].keys())[0]
    high_gold_label.append(label_dic[high_gold])
    low_gold_label.append(label_dic[low_gold])
    high_predict_label.append(label_dic[high_predict])
    low_predict_label.append(label_dic[low_predict])
    if high_predict == high_gold:
        high_correct += 1
    if low_predict == low_gold:
        low_correct += 1

print(num)
high_acc = high_correct / num
low_acc = low_correct / num
print("high ACC Score:", high_acc)
print("low ACC Score:", low_acc)
print("avg ACC Score:", (high_acc + low_acc)/2)
print("difference ACC Score:", high_acc - low_acc)

print(f"len high_ppl_list: {len(high_ppl_list)}")
print(f"len low_ppl_list: {len(low_ppl_list)}")

high_ppl = sum(high_ppl_list) / len(high_ppl_list)
low_ppl = sum(low_ppl_list) / len(low_ppl_list)
print(f"high_ppl: {high_ppl}")
print(f"low_ppl: {low_ppl}")
print(f"avg ppl: {(low_ppl + high_ppl)/2}")
print(f"difference ppl: {low_ppl - high_ppl}")

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

print(fail_count_high)
print(fail_count_low)
