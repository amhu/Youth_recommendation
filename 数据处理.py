import os
import re
import numpy as np
import jieba
import pickle
from keras.preprocessing import sequence   #sequence.pad_sequences,用于统一句子长度
from keras.utils import np_utils
jieba.load_userdict('/home/amhu/文档/sogou.txt')


'''读取stop停用词'''
def readstopwords(stopwords_path):
    stop_single_words=[]
    with open(stopwords_path,'r+',encoding='utf-8') as lines:
        for line in lines:
            content=line.strip()
            stop_single_words.append(content)
    return stop_single_words

'''把文档中的所有文件路劲存入数组'''
def getFilelist(data_paths):
    txtpaths = []
    for data_path in data_paths:
        pathDir = os.listdir(data_path)
        for allDir in pathDir:
            allDir=data_path+'/'+allDir
            if(allDir.find('txt') >= 0 and allDir.find('result') < 0):
                txtpaths.append(allDir)
    return txtpaths

'''对每个样本(文章)进行分词，去停用词'''
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



'''统计出所有样本的分词，建立样本词典'''
def wordscounter(data_paths):
    word_freqs = {}
    word2index = {}
    sample_length = []  # 保存所有样本数据的长度
    txtpaths = getFilelist(data_paths)
    n=0
    txts_result={}
    for txtpath in txtpaths:
        txt_result=sentense_cut(txtpath)
        txts_result[txtpath]=txt_result
        print('正在对'+str(n)+'篇文章进行分词')
        sample_length.append(len(txt_result))
        n+=1
        for word in txt_result:
            if word not in word_freqs.keys():
                word_freqs[word] = 1
            else:
                word_freqs[word] += 1
    ave_len = int(np.mean(sample_length))
    print(ave_len)
    key_value = sorted(word_freqs.items(), key=lambda v: v[1], reverse=True)
    word2index["PAD"] = 0;word2index["UNK"] = 1
    ID=2
    for key, value in key_value:
        word2index[key] = ID
        ID+=1
    return word2index,word_freqs,ave_len,txtpaths,txts_result


def word2digital(data_paths):
    word2index, word_freqs, MAX_SENTENCE_LENGTH,txtpaths,txts_result=wordscounter(data_paths)
    num_recs = len(txtpaths)
    X = np.empty(num_recs, dtype=list)
    Y = np.zeros(num_recs)
    for txtpath in txtpaths:
        for path in data_paths:
            if path in txtpath:
                Y[txtpaths.index(txtpath)] = data_paths.index(path)
        seqs=[]
        for word in txts_result[txtpath]:
            if word in word2index.keys():
                seqs.append(word2index[word])
            else:
                seqs.append(word2index["UNK"])
        X[txtpaths.index(txtpath)]=seqs

    X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH,padding='post')  # 统一句子长度，不够的补0，多出的截断
    num_classes=len(data_paths)
    Y =np_utils.to_categorical(Y, num_classes=num_classes)
    X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH, padding='post')  # 统一句子长度，不够的补0，多出的截断
    print(X)
    print(Y)
    return X, Y,word2index,MAX_SENTENCE_LENGTH


def write_to_file(content,file_name):
    with open(str(file_name)+'.txt', 'wb') as fw:
        pickle.dump(content, fw, -1)



stopwords_path='/home/amhu/文档/tycb_comment.txt'


data_paths=['/home/amhu/文档/多分类数据/财经','/home/amhu/文档/多分类数据/彩票','/home/amhu/文档/多分类数据/房产','/home/amhu/文档/多分类数据/股票','/home/amhu/文档/多分类数据/家居','/home/amhu/文档/多分类数据/社会', \
            '/home/amhu/文档/多分类数据/科技', '/home/amhu/文档/多分类数据/时政', '/home/amhu/文档/多分类数据/体育', '/home/amhu/文档/多分类数据/游戏',\
            '/home/amhu/文档/多分类数据/娱乐', '/home/amhu/文档/多分类数据/时尚']

X, Y,word2index,MAX_SENTENCE_LENGTH=word2digital(data_paths)
write_to_file(X,'X_特征向量(多分类)')
write_to_file(Y,'Y_结果向量(多分类)')
write_to_file(word2index,'分词ID词典(多分类)')
write_to_file(MAX_SENTENCE_LENGTH,'样本平均长度(多分类)')














