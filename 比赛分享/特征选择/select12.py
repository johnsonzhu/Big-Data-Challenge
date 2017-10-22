# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 16:10:42 2017

@author: Yang
"""

'''
特征选择
	wrapper02
	递归消除特征 也就是后项搜索
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import random
import xgboost

train = pd.read_csv(r"G:\比赛分享\data\alltrain.csv")
test= pd.read_csv(r"G:\比赛分享\data\alltest.csv")

#评分函数
#def score(y_pre, y_true):
#    
#    P = ((y_pre==0)&(y_true==0)).sum()/((y_pre==0).sum())
#    R = ((y_pre==0)&(y_true==0)).sum()/((y_true==0).sum())
#    
#    return (5*P*R)/(2*P+3*R)*100

#def score(y_pre):
#	num = 100000 - sum(y_pre)
#	
#	return num

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

#1. 特征的选择是随机的
#2. 设定失败搜索的次数
#3. 设定退出循环的阈值
#4. 设定最终特征的个数

num_fail = 10
random_index =  random.randint(0,len(feature)-1)
num_break = 18000
num_feature = 150

now_score = 0



while(len(feature)>num_feature):
	
	for i in range(num_fail):
		index = random.randint(0, len(feature)-1)
		drop_feature = feature[index]
		feature.remove(drop_feature)
		
		train_s = X[feature]
		test_s = test[feature]

		clf = xgboost.XGBClassifier() 
		clf.fit(train_s, y)
		restult = clf.predict(test_s)
		score = 100000 - sum(restult)
		
		print(i, score, drop_feature)
		if score > now_score:
			now_score = score
			print("up...")
			break
		elif i!=9:
			feature.append(drop_feature)
			print("the rest:" ,len(feature))
	print(' delete ',drop_feature)

file1 = open('feature03_houxiang.json','w',encoding='utf-8')  
json.dump(feature,file1, ensure_ascii=False)
file.close()

			

		
	
	
	
	
	
	
	
	
