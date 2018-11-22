#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 00:56:56 2018

@author: gabriel
"""
from scipy.stats import randint as sp_randint
from randomSearch import randomSearch
from sklearn.ensemble import RandomForestClassifier
#X_train, X_test, y_train, y_test, dummy1, dummy2 = processData()

def randomForest(X_train, X_test, y_train, search = False, n_iter_search = 100):
	if search:
		model = RandomForestClassifier()	
		
		param_dist = {'n_estimators':  sp_randint(200, 2000),
		    "max_depth": [3, None],
		    "max_features": [sp_randint(1, 11), "auto", "log2", None],
		    "min_samples_split": sp_randint(2, 11),
		    'min_samples_leaf': [1, 2, 4],
		    "bootstrap": [True, False],
		    "criterion": ["gini", "entropy"],
		    "n_jobs": [-1]}
		
		random_search = randomSearch(X_train, y_train, model, param_dist, n_iter_search)	
		
		y_pred = random_search.predict(X_test)
		
	else:
		model = RandomForestClassifier(bootstrap = False, criterion = 'gini', max_depth = None, max_features = 'auto', min_samples_leaf = 4, min_samples_split = 4, n_estimators = 508, n_jobs = -1)
		
		model.fit(X_train, y_train.values.ravel())
			
		y_pred = model.predict(X_test)
	
	return y_pred
