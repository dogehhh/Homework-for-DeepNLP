import jieba
import os
import matplotlib.pyplot as plt

# 读取中文停词
stop_file = 'cn_stopwords.txt'
with open(stop_file, 'r') as f:
    extra_characters = []
    lines = f.readlines()
    for line in lines:
        line = line[:-1]
        extra_characters.append(line)

# 读取中文语料库
txt_path = 'jyxstxtqj_downcc/'
counts = {}
for curdirpath, dirs, files in os.walk(txt_path, topdown=True):
    for file in files:
        if 'txt' in file:
            txt = open(txt_path + file, "r", encoding="gb18030").read()
            words = jieba.lcut(txt)
            for word in words:
                counts[word] = counts.get(word, 0) + 1
            print('{} succes'.format(file))

for word in extra_characters:
    if word in counts.keys():
        del counts[word]

items = list(counts.items())
items.sort(key=lambda x: x[1], reverse=True)
sort_list = sorted(counts.values(), reverse=True)


plt.title('Zipf-Law',fontsize=18)  #标题
plt.xlabel('rank',fontsize=18)     #排名
plt.ylabel('freq',fontsize=18)     #频度
plt.yticks([pow(10,i) for i in range(0,4)])  # 设置y刻度
plt.xticks([pow(10,i) for i in range(0,4)])  # 设置x刻度
x = [i for i in range(len(sort_list))]
plt.yscale('log')                  #设置纵坐标的缩放
plt.xscale('log')                  #设置横坐标的缩放
plt.plot(x, sort_list , 'r')       #绘图
plt.savefig('./Zipf_Law.jpg')      #保存图片
plt.show()
