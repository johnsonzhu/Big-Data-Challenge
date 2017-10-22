# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 21:41:24 2017

@author: Yang
"""

'''
特征选择
	wrapper12
	递归消除特征 也就是后项搜索
	但只支持 sklearn中带 .coef_  或 .feature_important_
'''
import pandas as pd
import json
from sklearn.feature_selection import RFE 
from sklearn.linear_model import LogisticRegression

train = pd.read_csv(r"G:\比赛分享\data\alltrain.csv")
test= pd.read_csv(r"G:\比赛分享\data\alltest.csv")

y = train['label']
del train['label']
del train['id']
id_a = test['id']
del test['id']




estimator = LogisticRegression()  # 逻辑斯谛回归
selector = RFE(estimator, 50, step=200)
selector = selector.fit(train, y)
selector.support_   # True的特征就是最终得到的特征

print(selector.ranking_)  # 值越小，越重要

new_train_data = train.ix[:, selector.support_]
new_test_data = test.ix[:, selector.support_]
print(new_train_data.shape)
print(new_test_data.shape)
feature12 = list(train.columns[selector.support_])
#保存
file1 = open('feature12.json','w',encoding='utf-8') 
json.dump(feature12,file1,ensure_ascii=False)  
file1.close() 



