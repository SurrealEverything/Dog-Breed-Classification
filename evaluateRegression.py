#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 22:20:05 2018

@author: gabriel
"""
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

def evaluateRegression(y_true, y_pred, model):

	print('\n', model.upper())
	
	expVar = explained_variance_score(y_true, y_pred)	
	print('Explained variance score: ', expVar)
	
	meanAbsVal = mean_absolute_error(y_true, y_pred)
	print('Mean absolute error: ', meanAbsVal)
	
	meanSqrErr = mean_squared_error(y_true, y_pred)
	print('Mean Squared Error: ', meanSqrErr)
	
	meanSqrLogErr = mean_squared_log_error(y_true, y_pred) 
	print('Mean Squared Log Error: ', meanSqrLogErr)
	
	medAbsVal = median_absolute_error(y_true, y_pred)
	print('Median absolute error: ', medAbsVal)
	
	r2Score = r2_score(y_true, y_pred) 
	print('R^2 score: ', r2Score)
	