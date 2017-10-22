# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 10:45:13 2017

@author: Yang

E-mail: xieear@qq.com
"""

import pandas as pd
import numpy as np
#from sklearn.externals.joblib import Parallel, delayed
import os
import warnings
warnings.filterwarnings("ignore")

#set path
path = r'J:\比赛\Mouse_tracking_recognition\data'
train_path = os.path.join(path, 'dsjtzs_txfz_training.txt')
test_path = os.path.join(path, 'dsjtzs_txfz_test1.txt')

#load data
train = pd.read_csv(train_path, sep=' ', names=['id','point', 'target','label']).ix[:1000]
#test = pd.read_csv(test_path, sep=' ', names=['id','point', 'target'])
label=train['label'].copy()
train.drop('label',axis=1,inplace=True)
#df = train.merge(test,how='outer')
df=train


def get_data(one):
    target_x = np.log1p(float(one.target.split(',')[0]))
    target_y = np.log1p(float(one.target.split(',')[1]))
    one['target_x'] = target_x
    one['target_y'] = target_y
    one['target'] = [target_x,target_y]
    
    points=[]
    points_x=[]
    points_y=[]
    points_t=[]
    
    dist_target=[]
    dist_x_target=[]
    dist_y_target=[]
    for point in one.point.split(';')[:-1]:
        point = point.split(',')
        x,y,t = np.log1p(float(point[0])), np.log1p(float(point[1])),np.log1p(float(point[2]))
        points_x.append(x)
        points_y.append(y)
        points_t.append(t)    
        points.append([x,y])
        
        dist_target.append(np.sqrt( (x-target_x)**2 + (y-target_y)**2 ))
        dist_x_target.append( x-target_x )
        dist_y_target.append( y-target_y )
        
    one['points'] = pd.Series(points)
    one['points_x'] = pd.Series(points_x)
    one['points_y'] = pd.Series(points_y)
    one['points_t'] = pd.Series(points_t) 
    one['count'] = one['points'].count()
    
    one['dist_target'] = pd.Series(dist_target)
    one['dist_x_target'] = pd.Series(dist_x_target)
    one['dist_y_target'] = pd.Series(dist_y_target)
    
    one['diff_x'] = one['points_x'].diff(1).dropna().reset_index(drop=True)
    one['diff_y'] = one['points_y'].diff(1).dropna().reset_index(drop=True)
    one['diff'] = np.sqrt( one['diff_x'].pow(2).add(one['diff_y'].pow(2)) )
    
    one['t_diff'] = one['points_t'].diff(1).dropna().reset_index(drop=True)
    if len(one['diff_x'])==0:
        one['diff_x'] = pd.Series([0])
        one['diff_y'] = pd.Series([0])
        one['diff'] = pd.Series([0])
        one['t_diff'] = pd.Series([1])
    
    
    one['v_x'] = one['diff_x'].div(one['t_diff'], fill_value = 0).replace({np.nan:0, np.inf:0, -np.inf:0})
    one['v_y'] = one['diff_y'].div(one['t_diff'], fill_value = 0).replace({np.nan:0, np.inf:0, -np.inf:0})
    one['v'] = one['diff'].div(one['t_diff'], fill_value = 0).replace({np.nan:0, np.inf:0, -np.inf:0})
    
    one['t_diff_2'] = one['t_diff'].diff(1).dropna().reset_index(drop=True)
    one['v_diff_x'] = one['v_x'].diff(1).dropna().reset_index(drop=True)
    one['v_diff_y'] = one['v_y'].diff(1).dropna().reset_index(drop=True)
    one['v_diff'] = one['v'].diff(1).dropna().reset_index(drop=True)
    if len(one['v_diff_x'])==0:
        one['v_diff_x'] = pd.Series([0])
        one['v_diff_y'] = pd.Series([0])
        one['v_diff'] = pd.Series([0])
        one['t_diff_2'] = pd.Series([1])
 
            
    one['a_x'] = one['v_diff_x'].div(one['t_diff_2'], fill_value=0).replace({np.nan:0, np.inf:0, -np.inf:0})
    one['a_y'] = one['v_diff_y'].div(one['t_diff_2'], fill_value=0).replace({np.nan:0, np.inf:0, -np.inf:0})
    one['a'] = one['v_diff'].div(one['t_diff_2'], fill_value=0).replace({np.nan:0, np.inf:0, -np.inf:0})
    return one.to_frame().T

def get_feature(data, name):
    dfGroup=pd.DataFrame()
    dfGroup[name+'_start'] = data.map(lambda x: x.ix[0])
    dfGroup[name+'_end'] = data.map(lambda x: x.ix[len(x)-1])
    dfGroup[name+'_max'] = data.map(pd.Series.max)
    dfGroup[name+'_min'] = data.map(pd.Series.min)
    dfGroup[name+'_range'] = dfGroup[name+'_max'].sub(dfGroup[name+'_min'])
    dfGroup[name+'_mean'] = data.map(pd.Series.mean)
    dfGroup[name+'_std'] = data.map(pd.Series.std)
    dfGroup[name+'_cv'] = dfGroup[name+'_std'].div(dfGroup[name+'_mean'], fill_value=0)
    dfGroup[name+'_Q1'] = data.map(lambda x: pd.Series.quantile(x, q=0.25))
    dfGroup[name+'_Q2'] = data.map(lambda x: pd.Series.quantile(x, q=0.5))
    dfGroup[name+'_Q3'] = data.map(lambda x: pd.Series.quantile(x, q=0.75))
    dfGroup[name+'_interRan'] = dfGroup[name+'_Q3'].sub(dfGroup[name+'_Q1'])
    dfGroup[name+'_skew'] = data.map(pd.Series.skew)
    dfGroup[name+'_kurt'] = data.map(pd.Series.kurt)
    
    return dfGroup

def get_point_feature():
    
    point_x = get_feature(df['points_x'], 'point_x')
    point_y = get_feature(df['points_y'], 'point_y')
    point = pd.concat([point_x, point_y], axis=1)
    
    point['point_target_x'] = df['target_x'].values
    point['point_target_y'] = df['target_y'].values
    
    point.to_csv(r'J:\比赛\Mouse_tracking_recognition\data\point.csv', index=None)
    return point
    
def get_dist_feature():
    dist_target = get_feature(df['dist_target'], 'dist_target')
    dist_x_target =  get_feature(df['dist_x_target'], 'dist_x_target')
    dist_y_target =  get_feature(df['dist_y_target'], 'dist_y_target')
    diff =  get_feature(df['diff'], 'diff')
    diff_x =  get_feature(df['diff_x'], 'diff_x')
    diff_y =  get_feature(df['diff_y'], 'diff_y')
    
    dist = pd.concat([dist_target, dist_x_target, dist_y_target,
                      diff, diff_x, diff_y], axis=1)
    dist.to_csv(r'J:\比赛\Mouse_tracking_recognition\data\dist.csv', index=None)
    return dist

def get_time_feature():
    t = get_feature(df['points_t'], 't')
    t_diff = get_feature(df['t_diff'], 't_diff')
    
    t = pd.concat([t, t_diff], axis=1)
    t.to_csv(r'J:\比赛\Mouse_tracking_recognition\data\t.csv', index=None)
    return t

def get_v_feature():
    v_x = get_feature(df['v_x'], 'v_x')
    v_y = get_feature(df['v_y'], 'v_y')
    v = get_feature(df['v'], 'v')
    v_diff_x = get_feature(df['v_diff_x'], 'v_diff_x')
    v_diff_y = get_feature(df['v_diff_y'], 'v_diff_y')
    v_diff = get_feature(df['v_diff'], 'v_diff')
    
    v = pd.concat([v_x, v_y, v,
                   v_diff_x, v_diff_y, v_diff], axis=1)
    v.to_csv(r'J:\比赛\Mouse_tracking_recognition\data\v.csv', index=None)
    return v
    
def get_a_feature():
    a_x = get_feature(df['a_x'], 'a_x')
    a_y = get_feature(df['a_y'], 'a_y')
    a = get_feature(df['a'], 'a')
    
    a = pd.concat([a_x, a_y, a], axis=1)
    a.to_csv(r'J:\比赛\Mouse_tracking_recognition\data\a.csv', index=None)
    return a

restult= (get_data(pd.Series(one)) for index,one in df.iterrows()) 
df= pd.concat(restult)
point = get_point_feature()
dist = get_dist_feature()
t = get_time_feature()
v = get_v_feature()
a = get_a_feature()

df1 = pd.concat([point, dist, t, v, a], axis=1)


'''
train = df.ix[:len(train)-1]
train['label']=label
test = df.ix[len(train):]

train.to_csv(r'J:\比赛\Mouse_tracking_recognition\data\train.csv',index=None)
test.to_csv(r'J:\比赛\Mouse_tracking_recognition\data\test.csv',index=None)
'''