# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 19:19:23 2017

@author: Yang
"""

'''
特征选择
	wrapper03
	前向搜索  往特征集合里添加特征
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import random
import xgboost
from sklearn.cross_validation import train_test_split

train = pd.read_csv(r"G:\比赛分享\data\alltrain.csv")
test= pd.read_csv(r"G:\比赛分享\data\alltest.csv")

#评分函数
def score(y_pre, y_true):
    
    P = ((y_pre==0)&(y_true==0)).sum()/((y_pre==0).sum())
    R = ((y_pre==0)&(y_true==0)).sum()/((y_true==0).sum())
    
    return (5*P*R)/(2*P+3*R)*100


y = train['label']
del train['label']
del train['id']
id_a = test['id']
del test['id']

fea = train.columns
#载入特征
file = open('feature01_lowVar.json','r',encoding='utf-8')  
feature = json.load(file) 
file.close()

#扩充数据
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train,y_train = smote.fit_sample(train,y)
X = pd.DataFrame(X_train, columns=fea)
y = pd.Series(y_train, name='label')


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

max_num = 20000
min_num = 10000
num_feature = 50

now_num = 0
now_score = 0

#now_feature = ['x_start']
now_feature = []
#feature.remove(now_feature[0])


while(len(now_feature)<num_feature):

	sf = pd.Series() #存储每次遍历特征的信息
	for i,f in enumerate(feature):
		now_feature.append(f)
		X1 = X_train[now_feature]
		X2 = X_val[now_feature]
		test1 = test[now_feature]
		
		clf = xgboost.XGBClassifier()
		clf.fit(X1, y_train)
		y_pre_val = clf.predict(X2)
		y_pre = clf.predict(test1)
		
		num = 100000 - sum(y_pre)
		score1 = score(y_pre_val,y_val)
		

		sf[f] = num
		now_feature.remove(f)
		
	sf = sf[sf<20000].sort_values(axis=0)

	if(sf[-1]>min_num and sf[-1]<max_num):
		now_feature.append(sf.index[-1])
		feature.remove(sf.index[-1])

		print("count:",sf[-1], "select",sf.index[-1])
		print( "num of feature:", len(now_feature))

	
		
		

file1 = open('feature03_qianxiang.json','w',encoding='utf-8')  
json.dump(now_feature,file1, ensure_ascii=False)
file1.close()
