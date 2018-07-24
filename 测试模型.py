import h5py
from keras.models import model_from_json
import re
import numpy as np
import jieba
import pickle
from keras.preprocessing import sequence   #sequence.pad_sequences,用于统一句子长度
from sklearn.model_selection import train_test_split

'''读取stop停用词'''
def readstopwords(stopwords_path):
    stop_single_words=[]
    with open(stopwords_path,'r+',encoding='utf-8') as lines:
        for line in lines:
            content=line.strip()
            stop_single_words.append(content)
    return stop_single_words

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

stopwords_path='G:/Youth_recommendation-master/哈工大停用词表.txt'
txt_path='G:/Youth_recommendation-master/test.txt'
word2index=read('分词ID词典(多分类5k)')
MAX_SENTENCE_LENGTH=read('样本平均长度(多分类5k)')
maxID=len(word2index)+1

# 读取、编译model
model=model_from_json(open('my_model_architecture5k.json').read())
model.load_weights('my_model_weights.h5')
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])



'''
#测试模型(选)
X=read('X_特征向量(多分类5k)')
Y=read('Y_结果向量(多分类5k)')
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.1, random_state=42)
loss,accuracy = model.evaluate(X_valid,Y_valid)
print(loss,accuracy)
'''

#使用模型分类
X=word2digital(txt_path,word2index,MAX_SENTENCE_LENGTH).reshape(1,MAX_SENTENCE_LENGTH)
y_pred = model.predict(X)
print('Network prediction:', np.argmax([y_pred[0]]))
topics_list=['财经','彩票','房产','股票','家居','社会','科技', '时政', '体育', '游戏', '娱乐', '时尚']
print('该文章属于%s类'% (topics_list[np.argmax([y_pred[0]])]))