# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 22:31:38 2017

@author: Yang
"""

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import Normalizer
'''
特征降维
	PCA
'''
import pandas as pd

train = pd.read_csv(r"G:\比赛分享\data\alltrain.csv")
test= pd.read_csv(r"G:\比赛分享\data\alltest.csv")

y = train['label']
del train['label']
del train['id']
id_a = test['id']
del test['id']

nor = Normalizer()
train = nor.fit_transform(train)
test = nor.transform(test)

lda = LDA(n_components=1)#分类标签数-1 就只能产生一维的特征
train_lda = lda.fit_transform(train, y)
test_lda = lda.transform(test)

pca = PCA(50)
train_pca = pca.fit_transform(train)
test_pca = pca.transform(test)



