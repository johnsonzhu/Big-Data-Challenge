# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 19:48:59 2017

@author: Yang
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import os
path = os.path.abspath(os.path.dirname(os.getcwd()))

train = pd.read_table(path + "\data\dsjtzs_txfz_training_sample.txt", sep=' ', names=['id','mouse_track','goal','label'])
#train = pd.read_table(path + "\data\dsjtzs_txfz_training.txt", sep=' ', names=['id','mouse_track','goal','label']).ix[2590:2610]
test = pd.read_table(path + "\data\dsjtzs_txfz_test_sample.txt", sep=' ', names=['id','mouse_track','goal',])


train['mouse_track'] = train['mouse_track'].apply(lambda x:[list(map(float,point.split(','))) for point in x.split(';')[:-1]])
train['goal'] = train['goal'].apply(lambda x: list(map(float,x.split(","))))



for i,data in train.iterrows():
	
	plt.figure(figsize=(16,9))	
	x = [point[0] for point in data['mouse_track']]
	y = [point[1] for point in data['mouse_track']]
	t = [point[2] for point in data['mouse_track']]

	plt.plot(x,y)
	plt.title('id {0}   label {1}'.format(i+1, data['label']))
	plt.savefig('{}.png'.format(i))
	
#	plt.scatter(x, y,c='r' )
#	plt.scatter(data['goal'][0], data['goal'][1],marker='*')
#	plt.title('id {0}   label {1}'.format(i+1, data['label']))
#	plt.savefig('带目标点 {}.png'.format(i))


