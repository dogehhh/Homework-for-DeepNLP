import jieba
from collections import Counter
import math
with open("预处理后的文本.txt", "r", encoding="utf-8") as f:
    corpus = [eve.strip("\n") for eve in f]

# 1-gram
token = []
for para in corpus:
    for word in para:
        token += word
token_num = len(token)
ct = Counter(token)
vocab1 = ct.most_common()
entropy_1gram = sum([-(eve[1]/token_num)*math.log((eve[1]/token_num),2) for eve in vocab1])
print("词库总词数：", token_num, " ", "不同词的个数：", len(vocab1))
print("出现频率前10的1-gram词语：", vocab1[:10])
print("entropy_1gram:", entropy_1gram)

# # 2-gram
# def combine2gram(cutword_list):
#     if len(cutword_list) == 1:
#         return []
#     res = []
#     for i in range(len(cutword_list)-1):
#         res.append(cutword_list[i] + "s"+ cutword_list[i+1])
#     return res
# token_2gram = []
# for para in corpus:
#     cutword_list = []
#     for word in para:
#         # cutword_list = jieba.lcut(para, cut_all=False)
#         cutword_list.append(word)
#     token_2gram += combine2gram(cutword_list)
# # 2-gram的频率统计
# token_2gram_num = len(token_2gram)
# ct2 = Counter(token_2gram)
# vocab2 = ct2.most_common()
# # print(vocab2[:20])
# # 2-gram相同句首的频率统计
# same_1st_word = [eve.split("s")[0] for eve in token_2gram]
# assert token_2gram_num == len(same_1st_word)
# ct_1st = Counter(same_1st_word)
# vocab_1st = dict(ct_1st.most_common())
# entropy_2gram = 0
# for eve in vocab2:
#     p_xy = eve[1]/token_2gram_num
#     first_word = eve[0].split("s")[0]
#     # p_y = eve[1]/vocab_1st[first_word]
#     entropy_2gram += -p_xy*math.log(eve[1]/vocab_1st[first_word], 2)
# print("词库总词数：", token_2gram_num, " ", "不同词的个数：", len(vocab2))
# print("出现频率前10的2-gram词语：", vocab2[:10])
# print("entropy_2gram:", entropy_2gram)

# # 3-gram
# def combine3gram(cutword_list):
#     if len(cutword_list) <= 2:
#         return []
#     res = []
#     for i in range(len(cutword_list)-2):
#         res.append(cutword_list[i] + cutword_list[i+1] + "s" + cutword_list[i+2] )
#     return res
# token_3gram = []
# for para in corpus:
#     cutword_list = []
#     for word in para:
#         # cutword_list = jieba.lcut(para, cut_all=False)
#         cutword_list.append(word)
#     # cutword_list = [removePunctuation(eve) for eve in cutword_list if removePunctuation(eve) != ""]
#     token_3gram += combine3gram(cutword_list)
# # 3-gram的频率统计
# token_3gram_num = len(token_3gram)
# ct3 = Counter(token_3gram)
# vocab3 = ct3.most_common()
# # print(vocab3[:20])
# # 3-gram相同句首两个词语的频率统计
# same_2st_word = [eve.split("s")[0] for eve in token_3gram]
# assert token_3gram_num == len(same_2st_word)
# ct_2st = Counter(same_2st_word)
# vocab_2st = dict(ct_2st.most_common())
# entropy_3gram = 0
# for eve in vocab3:
#     p_xyz = eve[1]/token_3gram_num
#     first_2word = eve[0].split("s")[0]
#     entropy_3gram += -p_xyz*math.log(eve[1]/vocab_2st[first_2word], 2)
# print("词库总词数：", token_3gram_num, " ", "不同词的个数：", len(vocab3))
# print("出现频率前10的3-gram词语：", vocab3[:10])
# print("entropy_3gram:", entropy_3gram)





