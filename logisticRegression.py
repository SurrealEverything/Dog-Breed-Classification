#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 19:42:30 2018

@author: gabriel
"""
from scipy.stats import randint as sp_randint
from randomSearch import randomSearch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import PolynomialFeatures

def logisticRegression(X_train, X_test, y_train, method = 1, degree = 3, n_iter_search = 1000):
	
	if degree > 1:
		poly = PolynomialFeatures(degree=3)
		X_train = poly.fit_transform(X_train)
		X_test = poly.fit_transform(X_test)
		
	if method == 0:
		model = LogisticRegression(tol = 0.0001, intercept_scaling = 0.9,
			solver = 'newton-cg', n_jobs = -1, multi_class = 'multinomial', 
			max_iter = 100, dual = False, C = 10000)
		
		model.fit(X_train, y_train.values.ravel())
	
		y_pred = model.predict(X_test)
		
	elif method == 1:
		logReg = LogisticRegression(tol = 0.0001, intercept_scaling = 0.9,
			solver = 'newton-cg', n_jobs = -1, multi_class = 'multinomial', 
			max_iter = 100, dual = False, C = 10000)
		
		model = BaggingClassifier(logReg, n_estimators = 100, max_samples=0.5, max_features=0.5, n_jobs = -1)
		
		model.fit(X_train, y_train.values.ravel())
	
		y_pred = model.predict(X_test)
		
	elif method == 2:
		model = LogisticRegression()
		
		param_dist = {"penalty": ['l1', 'l2'],
	              "dual": [True, False],
			  "tol": [0.0001, 0.00009, 0.00011, 0.0005, 0.00001, 0.001],
			  'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
			  'fit_intercept': [True, False],
			  'intercept_scaling': [1, 1.1, 2, 10, 0.5, 0.1, 0.9, 0, 1.2, 30, 3, 0.2, 0.01, 1.3, 0.08, 5, 15, 0.3, 0.2, 0.002],
			  'class_weight': ['balanced', None],
			  'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
			  'max_iter': [100, 105, 110, 140, 160, 180],
			  'multi_class': ['ovr', 'multinomial', 'auto'],
			  'n_jobs': [-1]}
		
		random_search = randomSearch(X_train, y_train, model, param_dist, n_iter_search)	
		
		y_pred = random_search.predict(X_test)
	
	else:
		logReg = LogisticRegression(tol = 0.0001, intercept_scaling = 0.9,
			solver = 'newton-cg', n_jobs = -1, multi_class = 'multinomial', 
			max_iter = 100, dual = False, C = 10000)
		model = BaggingClassifier(logReg)
		
		param_dist = {'n_estimators': sp_randint(10, 200),
		    'max_samples': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2],
		    'max_features': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2],
		    'bootstrap': [True, False],
		    'bootstrap_features': [True, False],
		    'oob_score': [True, False],
		    'n_jobs': [-1]}
		
		random_search = randomSearch(X_train, y_train, model, param_dist, n_iter_search)	
		
		y_pred = random_search.predict(X_test)	
		
	return y_pred 


