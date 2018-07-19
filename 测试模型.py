import h5py
from keras.models import model_from_json
import re
import numpy as np
import jieba
import pickle
from keras.preprocessing import sequence   #sequence.pad_sequences,用于统一句子长度
# jieba.load_userdict('F:/中国搜索项目数据/sogou.txt')

'''读取stop停用词'''
def readstopwords(stopwords_path):
    stop_single_words=[]
    with open(stopwords_path,'r+',encoding='utf-8') as lines:
        for line in lines:
            content=line.strip()
            stop_single_words.append(content)
    return stop_single_words

stopwords_path='F:/中国搜索项目数据/tycb_comment.txt'

def sentense_cut(txt_path):
    with open(txt_path, 'r+', encoding='utf-8') as lines:
        txt_result=[]
        for line in lines:
            sentense_result=[]
            line= re.sub("[\s+\d\.\!\/_,$%^*(+\"\']+|[+——！，。”“？、~@#￥%……&*（）]+", "", line)
            words=jieba.cut(line,cut_all=False)
            for word in words:
                if word != ' ' and word not in readstopwords(stopwords_path):
                    word_ = ''.join(word.split())
                    sentense_result.append(word_)
            txt_result.extend(sentense_result)
    return txt_result


def word2digital(txt_path,word2index,MAX_SENTENCE_LENGTH):
    txt_result=sentense_cut(txt_path)
    XX=[]
    X = np.empty(1, dtype=list)
    seqs=[]
    for word in txt_result:
        if word in word2index.keys():
            seqs.append(word2index[word])
        else:
            seqs.append(word2index["UNK"])
    X=seqs
    XX.append(X)
    XXX = sequence.pad_sequences(XX, maxlen=MAX_SENTENCE_LENGTH, padding='post')  # 统一句子长度，不够的补0，多出的截断
    return XXX

def read(file_name):
    with open(str(file_name)+'.txt', 'rb') as fr:
        data=pickle.load(fr)
    return  data

txt_path='F:/央广财经新闻18.txt'
word2index=read('分词ID词典(多分类)')
MAX_SENTENCE_LENGTH=read('样本平均长度(多分类)')
maxID=len(word2index)+1
print(MAX_SENTENCE_LENGTH)
# print(word2digital(txt_path,word2index,MAX_SENTENCE_LENGTH))

# 读取model
model=model_from_json(open('my_model_architecture.json').read())
model.load_weights('my_model_weights.h5')

X=word2digital(txt_path,word2index,MAX_SENTENCE_LENGTH).reshape(1,MAX_SENTENCE_LENGTH)
y_pred = model.predict(X)
print(y_pred)