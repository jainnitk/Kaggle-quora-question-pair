import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

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
    train = rebalance(train,target,0.17)
    model.fit(train[features],train[target])
    proba = list(zip(*model.predict_proba(valid[features])))[1]
    pd.DataFrame({"id":valid.index,tag1:proba})\
      .to_csv("train{0}_{1}.csv".format(tag1,tag2),index=False)
    print("logloss: {0}".format(log_loss(valid[target],proba)))
    proba = list(zip(*model.predict_proba(test)))[1]
    pd.DataFrame({"id":test.index,tag1:proba})\
      .to_csv("test{0}_{1}.csv".format(tag1,tag2),index=False)

def for_stack(train,test,model,tsize,rstate):
    features = list(train.columns.values)
    target = "is_duplicate"
    features.remove(target)
    
    train1, train2 = train_test_split(train,test_size=tsize,random_state=rstate)

    modeling(train1,train2,test,model,features,target,"R{0}".format(rstate),str(int(tsize*100))) 
    modeling(train2,train1,test,model,features,target,"R{0}".format(rstate),str(int((1-tsize)*100)))

##########################
# change the codes below #
##########################

from sklearn.ensemble import RandomForestClassifier

path = "../../features/03_team_features/"
train = pd.read_csv(path+"train_team.csv")
test = pd.read_csv(path+"test_team.csv")
model = RandomForestClassifier(100,n_jobs=4,verbose=True)

for_stack(train,test,model,tsize=0.45,rstate=10)
