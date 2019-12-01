#!/usr/bin/env python
# coding: utf-8

from defaults import *
import numpy as np

from keras.layers import Input, Dense, Embedding, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.core import Reshape, Flatten
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import Model,Sequential
from keras import regularizers

def one_hot_encod(labels):
    arr = []
    for l in labels:
        if(l==0):
            arr.append([0,1])
        else:
            arr.append([1,0])
    m = len(labels)
    arr = np.array(arr).reshape(m,2)
    return arr

#returns tokens
def remove_blank_tokens(tokens):
    non_blank = [w for w in tokens if w]
    return non_blank

def get_vocab(sent):
    sent_words = []
    for s in sent:
        sent_words.append(s)
    return sent_words

# Remove stopwords
def rem_sw(tokens):
    stop_w = set(sw.words('english'))
    # tokens = word_tokenize(text)
    res = [i for i in tokens if i not in stop_w]
    return res

#returns tokens
def word_tokenizer(text):
    # Removed punctuation and then split tokens on the basis of whitespaces
    # table = str.maketrans('', '', string.punctuation)
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) #map punctuation to space
    text = text.translate(translator)
    # text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    # tokens = [tok.translate(table) for tok in tokens]
    tokens = remove_blank_tokens(tokens)
    # Remove stop words
    tokens = rem_sw(tokens)
    return tokens

def read_file(file):
    fp = codecs.open(file, "r", encoding='utf-8', errors='replace')
    # text_w_meta = fp.read()
    lines = fp.readlines()    
    text_w_meta = [ l.lower() for l in lines]
    return text_w_meta
    

Name = "dataset"
file_lines= read_file(Name)
sentences = []
labels = []
t = str.maketrans("\n\t\r", "   ")
i =0
for l in file_lines:
    i+=1
    pair = l.split('\t')
    sz = len(pair)
    if(sz<2):
        print(pair)
        print(len(pair))
        print("i is",i)
    sentences.append(pair[0])
    pair[1] = pair[1].translate(t)
    labels.append(pair[1])
sent =[word_tokenizer(l) for l in sentences]
labels = [int(x) for x in labels]

#split into train and test
m = len(sent)
print("m is",m)
tr_size =  int(0.8*m)
test_size = m - tr_size
X_train = sent[:tr_size]
X_test = sent [tr_size:]
Y_train = labels[:tr_size]
Y_test =  labels[tr_size:]
print(len(X_train),len(Y_train))

vec_size = 300
vectors=[]
for sent in sent_words:
    sum = np.zeros([vec_size,1])
    for word in sent:
        vec = np.zeros([vec_size,1])            
        if word in model:
            vec = model[word]
            vec = np.reshape(vec,(vec_size,1))
        sum+=vec
    avg = sum/(len(sent))
    vectors.append(avg)
print(vectors[0])
        
X_tr=[]
Y_tr=[]
X_test=[]
Y_test=[]
for i in range(tr_size):
    X_tr.append(vectors[i])
    Y_tr.append(labels[i])
for i in range(tr_size,m):
    X_test.append(vectors[i])
    Y_test.append(labels[i])
# print(np.shape(X_tr))
# print(np.shape(Y_tr))
Y_tr = one_hot_encod(Y_tr)
Y_test = one_hot_encod(Y_test)
X_tr=np.array(X_tr)
X_test=np.array(X_test)
print(X_tr.shape,X_test.shape)
print(Y_tr.shape,Y_test.shape)

Y_tr=np.reshape(Y_tr,(tr_size,2))
Y_test=np.reshape(Y_test,(200,2))
print(np.shape(Y_test))
print(np.shape(X_tr))
model = Sequential()
model.add(Conv1D(200, kernel_size=2, activation='relu',batch_input_shape=(None,vec_size,1)))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x=X_tr, y=Y_tr,batch_size=32,epochs=3)
model.summary()

print(labels)

acc= model.evaluate(x=X_test,y=Y_test,batch_size=5)

print("Accuracy on test set",acc[1]) 