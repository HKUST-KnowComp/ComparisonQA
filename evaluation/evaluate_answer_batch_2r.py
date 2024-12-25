import json
import re

with open('../LLM_api/comparisonqa_answer_test_uncertainty_4omini_fewshot_total_output.json', 'r', encoding='utf-8') as f:
    predict_all = json.load(f)
    print(len(predict_all))

with open('../LLM_api/comparisonqa_answer_test_uncertainty_2r_4omini_fewshot_total_output.json', 'r', encoding='utf-8') as f:
    predict_2r_all = json.load(f)

with open('../comparisonqa_benchmark/comparisonqa_test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

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


num = 0
high_correct = 0
low_correct = 0
label_dic = {"A": 0, "B": 1, "C": 2, "D": 3, None: 4}

fail_count_high = 0
fail_count_low = 0
again_count_2r = {"high": 0, "low": 0}

for line in data:
    pattern = r'\*\*(.*?).\*\*'
    num += 1
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
        # if high_ppl < ppl_avg and high_predict == high_gold:
        #     high_correct += 1
        if (high_ppl <= ppl_avg and high_predict != high_gold) or (high_ppl >= ppl_avg and high_predict == high_gold):
            again_count_2r["high"] += 1
            assert line["question_id"]+"_high_A" in predict_2r_all
            high_predict = high_gold
            ans = []
            for o in ["A", "B", "C", "D"]:
                id = line["question_id"]+"_high_"+o
                a = predict_2r_all[id]
                if o == high_gold:
                    if "true" not in a.lower():
                        high_predict = None
                else:
                    if "true" in a.lower():
                        high_predict = None
                ans.append(a)
        if high_predict == high_gold:
            high_correct += 1
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
        # if low_ppl < ppl_avg and low_predict == low_gold:
        #     low_correct += 1
        if (low_ppl <= ppl_avg and low_predict != low_gold) or (low_ppl >= ppl_avg and low_predict == low_gold):
            again_count_2r["low"] += 1
            assert line["question_id"] + "_low_A" in predict_2r_all
            low_predict = low_gold
            ans = []
            for o in ["A", "B", "C", "D"]:
                id = line["question_id"] + "_low_" + o
                a = predict_2r_all[id]
                if o == low_gold:
                    if "true" not in a.lower():
                        low_predict = None
                else:
                    if "true" in a.lower():
                        low_predict = None
                ans.append(a)
        if low_predict == low_gold:
            low_correct += 1
    except:
        # print(predict_all[line["question_id"]+"_low"])
        fail_count_low += 1

print(num)
high_acc = high_correct / num
low_acc = low_correct / num
print("high ACC Score:", high_acc)
print("low ACC Score:", low_acc)
print("difference ACC Score:", high_acc - low_acc)

print(f"fail_count_high: {fail_count_high}")
print(f"fail_count_low: {fail_count_low}")
print(f"again_count_2r: {again_count_2r}")


