from collections import Counter

import jieba
import jieba.analyse
from wordcloud import WordCloud
from wordcloud import STOPWORDS


# Reference:
# https://tech.havocfuture.tw/blog/python-wordcloud-jieba
# https://hackmd.io/@aaronlife/python-bigdata-02
def generate_wordcloud(
    text,
    output_path,
    top_k=50,
    dictfile="data/wordcloud/dict.txt",
    stopfile="data/wordcloud/stopwords.txt",
    fontpath="data/wordcloud/msjh.ttc",
):
    jieba.set_dictionary(dictfile)
    jieba.analyse.set_stop_words(stopfile)

    tags = jieba.analyse.extract_tags(text, topK=top_k)

    seg_list = jieba.lcut(text, cut_all=False)
    dictionary = Counter(seg_list)

    freq = {}
    for ele in dictionary:
        if ele in tags:
            freq[ele] = dictionary[ele]

    wordcloud = WordCloud(
        background_color="white",
        margin=5,
        max_words=200,
        width=1280,
        height=720,
        font_path=fontpath,
    ).generate_from_frequencies(freq)

    wordcloud.to_file(output_path)
