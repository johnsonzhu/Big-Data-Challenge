# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 14:08:05 2017

@author: Yang
"""

'''
特征选择
	filter03
	利用xgb进行特征选择
'''
import pandas as pd
import json
import xgboost 
import matplotlib.pyplot as plt

train = pd.read_csv(r"G:\比赛分享\data\alltrain.csv")
y = train['label'].copy()
train = train.drop(['id','label'],axis=1)

xgb = xgboost.XGBClassifier()
xgb.fit(train, y)

feat_imp = pd.Series(xgb.booster().get_fscore()).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='xgb  Feature Importances')
plt.ylabel('Feature Importance Score')
plt.figure(figsize=(8,6))

feature3=list(feat_imp[feat_imp>2].index)

file1 = open('feature03.json','w',encoding='utf-8') 
json.dump(feature3,file1,ensure_ascii=False)  
file1.close() 

#file = open('feature01_lowVar.json','r',encoding='utf-8')  
#s = json.load(file) 
