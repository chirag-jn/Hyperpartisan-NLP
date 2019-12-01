# -*- coding: utf-8 -*-

import warnings
warnings.simplefilter("ignore", UserWarning)

import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, GlobalMaxPooling1D
import tensorflow as tf
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from defaults import *

dataset_loc = 'casedTrainedData'

vec_size = 768

train_vecs = np.zeros((Xy_train['Text_Token'].size, vec_size))

toks = Xy_train['Text_Token'].to_numpy()
for tok in range(toks.size):
  temp_vec = np.zeros((len(toks[tok]), vec_size))
  final_vec = np.zeros((1, vec_size))
  for t in range(len(toks[tok])):
    # print(toks[tok][t])
    if toks[tok][t] in word2vec:
      temp_vec[t] = word2vec[toks[tok][t]]
    else:
      temp_vec[t] = np.zeros((1, vec_size))
    final_vec = temp_vec.max(axis=0)
  train_vecs[tok] = np.array(final_vec)
print(train_vecs)
print(train_vecs.shape)

test_vecs = np.zeros((Xy_test['Text_Token'].size, vec_size), dtype='float32')

toks = Xy_test['Text_Token'].to_numpy()
for tok in range(toks.size):
  temp_vec = np.zeros((len(toks[tok]), vec_size))
  final_vec = np.zeros((1, vec_size))
  for t in range(len(toks[tok])):
    # print(toks[tok][t])
    if toks[tok][t] in word2vec:
      temp_vec[t] = word2vec[toks[tok][t]]
    else:
      temp_vec[t] = np.zeros((1, vec_size))
    final_vec = temp_vec.max(axis=0)
  test_vecs[tok] = np.array(final_vec)
print(test_vecs)
print(test_vecs.shape)

train_out = Xy_train[['Neg', 'Pos']].to_numpy()
test_out = Xy_test[['Neg', 'Pos']].to_numpy()

train_vecs = train_vecs[:, :, np.newaxis]
test_vecs = test_vecs[:, :, np.newaxis]


print('Train Inp Shape: ', train_vecs.shape)
print('Train Out Shape: ', train_out.shape)
print('Test Inp Shape: ', test_vecs.shape)
print('Test Out Shape: ', test_out.shape)

# train_vecs = tf.convert_to_tensor(train_vecs)

model = Sequential()

model.add(Conv1D(200, kernel_size=2, activation='relu', batch_input_shape=(None,vec_size,1)))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(y=train_out, x=train_vecs, epochs=3, batch_size=5)
model.summary()

acc = model.evaluate(x=test_vecs, y=test_out, batch_size=5)

print('Accuracy of the Model: ', acc[1])