#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 01:38:25 2018

@author: gabriel
"""
import numpy as np
from time import time
from sklearn.model_selection import RandomizedSearchCV

# Utility function to report best scores
def report(results, stop, n_top=1):
	for i in range(1, n_top + 1):
		candidates = np.flatnonzero(results['rank_test_score'] == i)
	for candidate in candidates:
		#print("Model with rank: {0}".format(i))
		print("\nMean validation score: {0:.3f} (std: {1:.3f})".format(
			results['mean_test_score'][candidate],
			results['std_test_score'][candidate]))
		print("Parameters: {0}".format(results['params'][candidate]))
		print("")
		
	print("RandomizedSearchCV took %.2f seconds." % stop)

def randomSearch(X_train, y_train, model, param_dist, n_iter_search):
	random_search = RandomizedSearchCV(model, param_distributions=param_dist,
			n_iter=n_iter_search, n_jobs=-1, cv=5, error_score = np.nan, verbose = 0)

	start = time()

	random_search.fit(X_train, y_train.values.ravel())

	stop = time() - start

	report(random_search.cv_results_, stop)
	
	writeParamsToFile(model.__class__.__name__, random_search.best_params_, stop)
	
	return random_search

def writeParamsToFile(modelName, paramsDt, stop):
	params = str(int(stop)) + ': ' + modelName + '('
	for key, val in paramsDt.items():
		params +=  key + ' = ' + str(val) + ', '
	params = params[:-2]
	params += ')\n'
	
	f = open("bestParams.txt", "a")
	f.write(params)