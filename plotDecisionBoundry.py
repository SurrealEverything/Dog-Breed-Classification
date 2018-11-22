#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 03:06:37 2018

@author: gabriel
"""
from sklearn.manifold.t_sne import TSNE
from sklearn.neighbors.classification import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

def plotDecisionBoundry(X, y, y_predicted, modelName):
	
	X_Train_embedded = TSNE(n_components=2).fit_transform(X)
	print (X_Train_embedded.shape)
	
	# create meshgrid
	resolution = 1000 # 100x100 background pixels
	X2d_xmin, X2d_xmax = np.min(X_Train_embedded[:,0]), np.max(X_Train_embedded[:,0])
	X2d_ymin, X2d_ymax = np.min(X_Train_embedded[:,1]), np.max(X_Train_embedded[:,1])
	xx, yy = np.meshgrid(np.linspace(X2d_xmin, X2d_xmax, resolution), np.linspace(X2d_ymin, X2d_ymax, resolution))
	
	# approximate Voronoi tesselation on resolution x resolution grid using 1-NN
	background_model = KNeighborsClassifier(n_neighbors=1).fit(X_Train_embedded, y_predicted) 
	voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
	voronoiBackground = voronoiBackground.reshape((resolution, resolution))
	
	#plot
	plt.contourf(xx, yy, voronoiBackground)
	plt.scatter(X_Train_embedded[:,0], X_Train_embedded[:,1], c=y.values.flatten())
	plt.title(modelName)
	plt.show()