from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import word2vec

sentences = word2vec.LineSentence('output/笑傲江湖_segment.txt')
# 训练语料
path = get_tmpfile("word2vec.model")  # 创建临时文件
model = word2vec.Word2Vec(sentences, hs=1, min_count=1, window=5, vector_size=200, sg=1, epochs=20)
# model.save("word2vec.model")

req_count = 10
for key in model.wv.similar_by_word('令狐冲', topn =100):
    if len(key[0]) > 1:
        req_count -= 1
        print(key[0], key[1])
        if req_count == 0:
            break

