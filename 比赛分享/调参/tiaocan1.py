# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 19:28:57 2017

@author: Yang
"""
'''调参
	1. 理解模型
	2. 列出所有的参数
	3. 选择对模型提升大的参数

	代码错误：
		1.     kstep = len(randidx) / nfold  改为  kstep = len(randidx) // nfold
		2.     'Disbursed'  改为 target
		3,     Parameter values should be a list.  改为  param_test1 = {'max_depth':list(range(3,10,2)),'min_child_weight':list(range(1,6,2))}
	
'''
#Import libraries:
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


train = pd.read_csv(r"G:\比赛分享\data\alltrain.csv")
test= pd.read_csv(r"G:\比赛分享\data\alltest.csv")

target = 'label'
IDcol = 'id'

def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
								
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics=['auc'], early_stopping_rounds=early_stopping_rounds, show_progress=True)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print ("\nModel Report")
    print (("Accuracy : %.4g") % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    print (("AUC Score (Train): %f" )% metrics.roc_auc_score(dtrain[target], dtrain_predprob))
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='xgb  Feature Importances')
    plt.ylabel('Feature Importance Score')

				
'''
一 修正用于调整基于树的参数的学习速率和估计量数  
	也就是 learning_rate n_estimators 学习速率和树的数量
'''
##Choose all predictors except target & IDcols
predictors = [x for x in train.columns if x not in [target, IDcol]]

xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, train, predictors)

###Step 2: Tune max_depth and min_child_weight
#param_test1 = {
# 'max_depth':list(range(3,10,2)),
# 'min_child_weight':list(range(1,6,2))
#}
#gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
# min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
# objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
# param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False,   cv=2   )
#
#print(gsearch1.fit(train[predictors],train[target]))
#print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)


#
#param_test2 = {
# 'max_depth':[4,5,6],
# 'min_child_weight':[4,5,6]
#}
#gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=5,
# min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
# objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
# param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#print(gsearch2.fit(train[predictors],train[target]))
#print(gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_)



###Step 3: Tune gamma
#param_test3 = {
# 'gamma':[i/10.0 for i in range(0,5)]
#}
#gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=4,
# min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
# objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
# param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#print(gsearch3.fit(train[predictors],train[target]))
#print(gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_)
##
#xgb2 = XGBClassifier(
# learning_rate =0.1,
# n_estimators=1000,
# max_depth=4,
# min_child_weight=6,
# gamma=0,
# subsample=0.8,
# colsample_bytree=0.8,
# objective= 'binary:logistic',
# nthread=4,
# scale_pos_weight=1,
# seed=27)
#modelfit(xgb2, train, predictors)
#

###Step 4: Tune subsample and colsample_bytree
#param_test4 = {
# 'subsample':[i/10.0 for i in range(6,10)],
# 'colsample_bytree':[i/10.0 for i in range(6,10)]
#}
#gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
# min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
# objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
# param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#print(gsearch4.fit(train[predictors],train[target]))
#print(gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_)
#
#param_test5 = {
# 'subsample':[i/100.0 for i in range(75,90,5)],
# 'colsample_bytree':[i/100.0 for i in range(75,90,5)]
#}
#gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
# min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
# objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
# param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#gsearch5.fit(train[predictors],train[target])


###Step 5: Tuning Regularization Parameters
#param_test6 = {
# 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
#}
#gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
# min_child_weight=6, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
# objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
# param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#print(gsearch6.fit(train[predictors],train[target]))
#print(gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_)

###Step 6: Reducing Learning Rate
#xgb4 = XGBClassifier(
# learning_rate =0.01,
# n_estimators=5000,
# max_depth=4,
# min_child_weight=6,
# gamma=0,
# subsample=0.8,
# colsample_bytree=0.8,
# reg_alpha=0.005,
# objective= 'binary:logistic',
# nthread=4,
# scale_pos_weight=1,
# seed=27)
#modelfit(xgb4, train, predictors)

