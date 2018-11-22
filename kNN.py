#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 01:09:22 2018

@author: gabriel
"""
from scipy.stats import randint as sp_randint
from randomSearch import randomSearch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

def kNN(X_train, X_test, y_train, method =0, n_iter_search = 1000):
	
	if method == 0:
		model = KNeighborsClassifier(algorithm = 'kd_tree', leaf_size = 40, n_neighbors = 4, p = 2, weights = 'distance')
		
		model.fit(X_train, y_train.values.ravel())
			
		y_pred = model.predict(X_test)
	
	elif  method == 1:
		knn = KNeighborsClassifier(algorithm = 'kd_tree', leaf_size = 40, n_neighbors = 4, p = 2, weights = 'distance')
	
		model = BaggingClassifier(knn, n_estimators = 100, max_samples=0.5, max_features=0.5, n_jobs = -1)
		
		model.fit(X_train, y_train.values.ravel())
	
		y_pred = model.predict(X_test)
		
	elif method == 2:
		model = KNeighborsClassifier()	
		
		param_dist = {"n_neighbors": list(range(1, 31)),
		    "p": sp_randint(1, 4), 
		    "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
		    "weights": ['uniform', 'distance'],
		    "leaf_size": [10, 20, 25, 28, 30, 32, 35, 40, 60],
		    "n_jobs": [-1]}
		
		random_search = randomSearch(X_train, y_train, model, param_dist, n_iter_search)	
		
		y_pred = random_search.predict(X_test)
		
	else:
		knn = KNeighborsClassifier(algorithm = 'kd_tree', leaf_size = 40, n_neighbors = 4, p = 2, weights = 'distance')
		
		model = BaggingClassifier(knn)
		
		param_dist = {'n_estimators': sp_randint(10, 200),
		    'max_features': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2],
		    'bootstrap': [True, False],
		    'bootstrap_features': [True, False],
		    'oob_score': [True, False],
		    'n_jobs': [-1]}
		
		random_search = randomSearch(X_train, y_train, model, param_dist, n_iter_search)	
		
		y_pred = random_search.predict(X_test)
		
	return y_pred
