#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 19:14:59 2018

@author: gabriel
"""

from sklearn import linear_model
from randomSearch import randomSearch
from scipy.stats import randint as sp_randint
import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import PolynomialFeatures

def linearRegression(X_train, X_test, y_train, method = 1, degree = 1, n_iter_search = 100):

	
	if degree > 1:
		poly = PolynomialFeatures(degree)
		X_train = poly.fit_transform(X_train)
		X_test = poly.fit_transform(X_test)
	
	if method == 0:
		model = linear_model.LinearRegression()
		
		model.fit(X_train, y_train)
		
		y_pred = model.predict(X_test)

	elif method == 1:
		model = linear_model.Ridge()
		
		model.fit(X_train, y_train)
		
		y_pred = model.predict(X_test)
	
	elif method == 2:
		model = linear_model.Lasso(alpha = 0.0002704959730463137, fit_intercept = True, max_iter = 1698, normalize = True, positive = False, selection = 'random', tol = 0.001)
		
		model.fit(X_train, y_train)
		
		y_pred = model.predict(X_test)
	
	elif method == 3:
		model = linear_model.ElasticNet(alpha = 0.0002465811075822604, fit_intercept = True, l1_ratio = 0.9, max_iter = 1508, normalize = True, positive = False, selection = 'random', tol = 0.001)
		
		model.fit(X_train, y_train)
		
		y_pred = model.predict(X_test)
	elif method == 4:
		model = linear_model.Ridge()
		
		alphas = np.logspace(-10, -2, 200)
		
		param_dist = {'alpha': alphas,
		    'fit_intercept': [True, False],
		    'normalize': [True, False],
		    'max_iter': sp_randint(500, 2000),
		    "tol": [0.0001, 0.00009, 0.00011, 0.0005, 0.00001, 0.001],
		    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
		
		random_search = randomSearch(X_train, y_train, model, param_dist, n_iter_search)
		
		y_pred = random_search.predict(X_test)
	elif method == 5:
		model = linear_model.Lasso()
		
		alphas = np.logspace(-10, -2, 200)
		
		param_dist = {'alpha': alphas,
		    'fit_intercept': [True, False],
		    'normalize': [True, False],
		    'max_iter': sp_randint(500, 2000),
		    "tol": [0.0001, 0.00009, 0.00011, 0.0005, 0.00001, 0.001],
		    'positive': [True, False],
		    'selection': ['random', 'cyclic']}
		
		random_search = randomSearch(X_train, y_train, model, param_dist, n_iter_search)
		
		y_pred = random_search.predict(X_test)
		
	elif method == 6:
		model = linear_model.ElasticNet()
		
		alphas = np.logspace(-10, -2, 200)
		
		param_dist = {'alpha': alphas,
		    'l1_ratio': [0, 0.1, 0.2, 0.25, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.75, 0.8, 0.9, 1],
		    'fit_intercept': [True, False],
		    'normalize': [True, False],
		    'max_iter': sp_randint(500, 2000),
		    "tol": [0.0001, 0.00009, 0.00011, 0.0005, 0.00001, 0.001],
		    'positive': [True, False],
		    'selection': ['random', 'cyclic']}
		
		random_search = randomSearch(X_train, y_train, model, param_dist, n_iter_search)
		
		y_pred = random_search.predict(X_test)
		
	elif method == 7:
		linReg = linear_model.ElasticNet(alpha = 0.0002465811075822604, fit_intercept = True, l1_ratio = 0.9, max_iter = 1508, normalize = True, positive = False, selection = 'random', tol = 0.001)
		
		model = BaggingRegressor(linReg, n_jobs = -1)
		
		model.fit(X_train, y_train.values.ravel())
	
		y_pred = model.predict(X_test)
		
	return y_pred
	