#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 23:44:06 2017

@author: zhouyu
"""

import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding,GlobalMaxPooling1D
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Input, Embedding, LSTM, Concatenate
from keras.layers.normalization import BatchNormalization

train_data = pd.read_csv("train.csv")
train_data.fillna("",inplace = True)
test_data = pd.read_csv("test.csv")

# optimal clean the words before start 
# clean the text,copied from https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings
def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.
    # Convert words to lower case and split them
    try:
        text = text.lower().split()
    except:
        return ""
    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    text = " ".join(text)
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return text

train_data['question1'] = train_data.apply(lambda row:text_to_wordlist(row['question1']), axis = 1)
train_data['question2'] = train_data.apply(lambda row:text_to_wordlist(row['question2']), axis = 1)
test_data['question1'] = test_data.apply(lambda row:text_to_wordlist(row['question1']), axis = 1)
test_data['question2'] = test_data.apply(lambda row:text_to_wordlist(row['question2']), axis = 1)
# turn text into sequence:
X_train_1 = train_data['question1'].values.astype(str)
X_train_2 = train_data['question2'].values.astype(str)
X_test_1 = test_data["question1"].values.astype(str)
X_test_2 = test_data["question2"].values.astype(str)

X_raw = np.hstack((X_train_1,X_train_2,X_test_1,X_test_2))

tokenizer = Tokenizer(num_words = MAX_WORDS,lower = True)
tokenizer.fit_on_texts(X_raw)
sequences = tokenizer.texts_to_sequences(X_raw)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

l1 = X_train_1.shape[0]
l2 = X_train_2.shape[0]+l1
l3 = X_test_1.shape[0]+l2

x_data = pad_sequences(sequences, maxlen = MAX_SEQUENCE)
x1 = x_data[:l1]
x2 = x_data[l1:l2]
x3 = x_data[l2:l3]
x4 = x_data[l3:]

# building keras models
hidden_dims = 250
filters = 25
kernel_size = 3
hidden_dims = 250
epochs = 64
batch_size = 32
hidden_dims2 = 32
lstm_output_size = 70

## model adopt from Kaggle
epochs = 2
num_lstm = np.random.randint(100, 128)
num_dense = np.random.randint(100, 150)
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25
lstm_layer = LSTM(num_lstm,dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

input1 = Input(shape = x1.shape[1:],dtype = 'float32',name = 'input1')
branch1 = embed_layer(input1)
branch1 = lstm_layer(branch1)
#branch1 = Conv1D(filters, kernel_size, padding='valid', activation='relu',strides=1)(input1)
#branch1 = GlobalMaxPooling1D()(branch1)
input2 = Input(shape = x2.shape[1:],dtype = 'float32',name = 'input2')
branch2 = embed_layer(input2)
branch2 = lstm_layer(branch2)
mergeone = concatenate([branch1, branch2],axis = -1)
mergeone = Dropout(rate_drop_dense)(mergeone)
mergeone = BatchNormalization()(mergeone)
mergeone = Dense(hidden_dims, activation = 'relu')(mergeone)
mergeone = Dropout(rate_drop_dense)(mergeone)
mergeone = BatchNormalization()(mergeone)
m_input = Dense(hidden_dims2,activation = 'relu',name = 'm_input')(mergeone)
main_output = Dense(1,activation = 'sigmoid',name='main_output')(m_input)
modelZ = Model(inputs=[input1, input2], outputs= main_output)
modelZ.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
modelZ.fit([x1,x2], y_train,
          batch_size=1024,
          epochs=epochs,verbose = 1,
          validation_split = 0.2,shuffle = True)

## get intermediate input
layer_name = 'm_input'
intermediate_layer_model = Model(inputs = modelZ.input,
                                 outputs = modelZ.get_layer(layer_name).output)
intermediate_train = intermediate_layer_model.predict([x1,x2], 
                                                      batch_size = 2048, verbose = 1)
intermediate_test = intermediate_layer_model.predict([x3,x4],
                                                     batch_size = 2048, verbose = 1)

# save intermediate output
cols = np.arange(hidden_dims2)+1
cols = cols.astype(str)
cols =np.core.defchararray.add('lstm_',cols) 
lstm_train = pd.DataFrame(intermediate_train, columns = cols)
lstm_test = pd.DataFrame(intermediate_test, columns = cols)
lstm_train.to_csv("lstm_embed_train.csv")
lstm_test.to_csv("lstm_embed_test.csv")