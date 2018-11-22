#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 21:38:03 2018

@author: gabriel
"""
from sklearn.neighbors import KNeighborsRegressor
from randomSearch import randomSearch
from scipy.stats import randint as sp_randint
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import PolynomialFeatures

def kNNRegression(X_train, X_test, y_train, method = 0, degree = 1, n_iter_search = 100):
	
	if degree > 1:
		poly = PolynomialFeatures(degree=3)
		X_train = poly.fit_transform(X_train)
		X_test = poly.fit_transform(X_test)
	
	if method == 0:
		model = KNeighborsRegressor(algorithm = 'ball_tree', leaf_size = 28, n_jobs = -1, n_neighbors = 22, p = 1, weights = 'uniform')
		
		model.fit(X_train, y_train)
		
		y_pred = model.predict(X_test)
	
	if method == 1:
		model = KNeighborsRegressor()	
		
		param_dist = {"n_neighbors": list(range(1, 31)),
		    "p": sp_randint(1, 4), 
		    "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
		    "weights": ['uniform', 'distance'],
		    "leaf_size": [10, 20, 25, 28, 30, 32, 35, 40, 60],
		    "n_jobs": [-1]}
		
		random_search = randomSearch(X_train, y_train, model, param_dist, n_iter_search)	
		
		y_pred = random_search.predict(X_test)
	
	elif method == 2:
		knnReg = KNeighborsRegressor(algorithm = 'ball_tree', leaf_size = 28, n_jobs = -1, n_neighbors = 22, p = 1, weights = 'uniform')
		
		model = BaggingRegressor(knnReg, n_jobs = -1)
		
		model.fit(X_train, y_train.values.ravel())
	
		y_pred = model.predict(X_test)	
	return y_pred 