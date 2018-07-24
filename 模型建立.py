import os
import keras
from keras.utils import np_utils
from keras.models import Sequential #序贯模型,用来一层一层一层的去建立神经层
from keras.layers import Dense, Activation,Embedding,LSTM,Bidirectional#全连接层和激活层
from keras.optimizers import RMSprop
from keras import regularizers
from sklearn.model_selection import train_test_split
import numpy as np
import json
import pickle
from keras.callbacks import EarlyStopping
import h5py
from keras.models import model_from_json
import matplotlib.pyplot as plt


def read(file_name):
    with open(str(file_name)+'.txt', 'rb') as fr:
        data=pickle.load(fr)
    return  data

X=read('X_特征向量(多分类1w)')
Y=read('Y_结果向量(多分类1w)')
word2index=read('分词ID词典(多分类1w)')
MAX_SENTENCE_LENGTH=read('样本平均长度(多分类1w)')
maxID=len(word2index)+1



def model(X,Y,MAX_SENTENCE_LENGTH,vocab_size):
    epochs_list = []
    X_train_valid, X_test, Y_train_valid, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_valid, Y_train_valid, test_size=0.1, random_state=42)

    EMBEDDING_SIZE = 200
    HIDDEN_LAYER_SIZE = 48
    BATCH_SIZE = 500
    NUM_EPOCHS = 8
    for i in range(NUM_EPOCHS):
        epochs_list.append(i + 1)
    model = Sequential()
    model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH))
    model.add(Bidirectional(LSTM(HIDDEN_LAYER_SIZE, dropout=0.3, recurrent_dropout=0.3)))
    model.add(Dense(12))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,validation_data=(X_valid, Y_valid),callbacks=[EarlyStopping])
    score, acc = model.evaluate(X_test, Y_test)
    return score, acc, epochs_list, history, model, X_test, Y_test





def draw(epochs_list, history):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(epochs_list, history.history['loss'], 'bo')
    plt.plot(epochs_list, history.history['loss'])
    plt.plot(epochs_list, history.history['val_loss'], 'rs')
    plt.plot(epochs_list, history.history['val_loss'])

    ax.set_title('model train vs validation loss')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.grid(True)
    plt.show()

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
    plt.plot(epochs_list, history.history['acc'], 'bo')
    plt.plot(epochs_list, history.history['acc'])
    plt.plot(epochs_list, history.history['val_acc'], 'rs')
    plt.plot(epochs_list, history.history['val_acc'])

    ax1.set_title('model train vs validation acc')
    ax1.set_ylabel('acc')
    ax1.set_xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.grid(True)
    plt.show()



EarlyStopping=EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
score, acc,epochs_list,history,model,Xtest,ytest=model(X,Y,MAX_SENTENCE_LENGTH,maxID)

model.save('my_model.h5')   # HDF5 file, you have to pip3 install h5py if don't have it
json_string = model.to_json()
with open('my_model_architecture5k.json','w') as f:
    f.write(json_string)
model.save_weights('my_model_weights.h5')

print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))
draw(epochs_list, history)






