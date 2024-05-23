import jieba
import jieba.analyse



jieba.suggest_freq('令狐冲', True)
jieba.suggest_freq('岳不群', True)
jieba.suggest_freq('林平之', True)
jieba.suggest_freq('岳灵珊', True)
jieba.suggest_freq('田伯光', True)
jieba.suggest_freq('岳夫人', True)
jieba.suggest_freq('任盈盈', True)


with open('jyxstxtqj_downcc/笑傲江湖.txt',encoding='gb18030') as f:
    document = f.read()

    # document_decode = document.decode('GBK')

    document_cut = jieba.cut(document)
    # print  ' '.join(jieba_cut)  //如果打印结果，则分词效果消失，后面的result无法显示
    result = ' '.join(document_cut)
    with open('output/笑傲江湖_segment.txt', 'w',encoding="utf-8") as f2:
        f2.write(result)



