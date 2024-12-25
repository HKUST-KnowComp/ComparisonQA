import json

with open('../LLM_api/longtailqa_test_perplexity_Meta_Llama_3_8B_Instruct.json', 'r', encoding='utf-8') as f:
    perplexity_all = json.load(f)

# print(len(perplexity_all))
# print(perplexity_all[-1])

label_dic = {"A": 0, "B": 1, "C": 2, "D": 3}
high_ppl_list = []
low_ppl_list = []
perplexity_high_avg = 0
perplexity_low_avg = 0
for line in perplexity_all:
    high_gold = list(line["high_question"]["answer"].keys())[0]
    low_gold = list(line["low_question"]["answer"].keys())[0]
    high_p = line["high_question"]["perplexity_gen"][label_dic[high_gold]]
    low_p = line["low_question"]["perplexity_gen"][label_dic[low_gold]]
    perplexity_high_avg += high_p
    perplexity_low_avg += low_p
    high_ppl_list.append(high_p)
    low_ppl_list.append(low_p)

perplexity_high_avg = perplexity_high_avg / len(perplexity_all)
perplexity_low_avg = perplexity_low_avg / len(perplexity_all)

print(perplexity_high_avg)
print(perplexity_low_avg)
print(perplexity_low_avg - perplexity_high_avg)
