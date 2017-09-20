import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from keras.regularizers import l2
from keras import regularizers
from keras import callbacks
from sklearn import preprocessing

def rebalance(df,ylabel,target_pos_ratio):
    y = df[ylabel]
    total_num = len(y)
    pos_num = sum(y)
    neg_num = total_num-pos_num
    current_pos_ratio = 1.0*pos_num/total_num
    if target_pos_ratio >= current_pos_ratio:
        n = int(neg_num/(1-target_pos_ratio)-neg_num/(1-current_pos_ratio))
        pos_boostrap_sample = df[df[ylabel] == 1].sample(n, replace = True)
    elif target_pos_ratio < current_pos_ratio:
        n = int(pos_num/target_pos_ratio-pos_num/current_pos_ratio)
        pos_boostrap_sample = df[df[ylabel] == 0].sample(n, replace = True)
    return pd.concat((pos_boostrap_sample, df))


def modeling(train,valid,test,model,features,target,tag1,tag2):
    path = "data/train/"
    train = rebalance(train,target,0.17)
    model.fit(train[features].values,train[target].values,epochs = 20,batch_size = 64,verbose = 2,
              callbacks = [earlyStopping],
              validation_split = 0.2,shuffle = True)
    #proba = list(zip(*model.predict(valid[features].values)))[1]
    print "evaluate on valid set:"
    print model.evaluate(valid[features].values,valid[target].values)
    proba = model.predict(valid[features].values).reshape((-1))
    pd.DataFrame({"id":valid.index,tag1:proba})\
        .to_csv(path+"sadahanu1_train{0}_{1}.csv".format(tag1,tag2),index=False)
    #proba = list(zip(*model.predict(test.values)))[1]
    #print "evaluate on valid set:"
    proba = model.predict(test[features].values).reshape((-1))
    pd.DataFrame({"id":test.index,tag1:proba})\
        .to_csv(path+"sadahanu1_test{0}_{1}.csv".format(tag1,tag2),index=False)

def for_stack(train,test,model1,model2,tsize,rstate):
    features = list(train.columns.values)
    target = "is_duplicate"
    features.remove(target)
    train1, train2 = train_test_split(train,test_size=tsize,random_state=rstate)
    #print train2.shape
    #return train2
    modeling(train1,train2,test,model1,features,target,"R{0}".format(rstate),str(int(tsize*100)))
    modeling(train2,train1,test,model2,features,target,"R{0}".format(rstate),str(int((1-tsize)*100)))

##########################
# change the codes below #
##########################

path = "data/train/"
train = pd.read_csv(path+"train_103.csv")
tests = pd.read_csv(path+"test_103.csv")

#insert two layer keras model
model1 = Sequential()
model1.add(Dense(256,input_dim = train.shape[1]-1,
                kernel_regularizer=regularizers.l2(1e-6),
                kernel_initializer = 'he_uniform'))
model1.add(Activation('relu'))
model1.add(Dropout(0.1))
model1.add(Dense(64,activation = 'relu'))
model1.add(Dropout(0.3))
model1.add(Dense(1,activation = 'sigmoid'))

model1.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
earlyStopping=callbacks.EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')
## model end here
# navid model2
model2 = Sequential()
model2.add(Dense(256,input_dim = train.shape[1]-1,
                 kernel_regularizer=regularizers.l2(1e-6),
                 kernel_initializer = 'he_uniform'))
model2.add(Activation('relu'))
model2.add(Dropout(0.1))
model2.add(Dense(64,activation = 'relu'))
model2.add(Dropout(0.3))
model2.add(Dense(1,activation = 'sigmoid'))

model2.compile(loss='binary_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])
for_stack(train,test,model1,model2,tsize=0.45,rstate=10)
