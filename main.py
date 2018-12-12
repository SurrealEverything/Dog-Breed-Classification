#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 19:40:44 2018

@author: gabriel
"""

from dataPreprocessing import processData
from logisticRegression import logisticRegression
from randomForest import randomForest
from kNN import kNN
from linearRegression import linearRegression
from kNNRegression import kNNRegression
from evaluateClassification import evaluateClassification
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from analyseData import analyseData
from plotDecisionBoundry import plotDecisionBoundry
from evaluateRegression import evaluateRegression
from kmeans import kmeans
from dbscan import dbscan
from agglomerative import agglomerative
from evaluateClustering import evaluateClustering
from vizualizeData import vizualizeData


#loading pre-processed data
X_train, X_test, yClass_train, yClass_test, yReg_train, yReg_test = processData()

yClass_lr = logisticRegression(X_train, X_test, yClass_train)
lrAc = accuracy_score(yClass_test, yClass_lr)
X_lr = X_test

#reloading data pre-processed with different parameters
X_train, X_test, yClass_train, yClass_test, yReg_train, yReg_test = processData(normalization = 'mms')

yClass_rf = randomForest(X_train, X_test, yClass_train)
rfAc = accuracy_score(yClass_test, yClass_rf)
X_rf = X_test

yClass_knn = kNN(X_train, X_test, yClass_train)
knnAc = accuracy_score(yClass_test, yClass_knn)
X_knn = X_test

X_train, X_test, yClass_train, yClass_test, yReg_train, yReg_test = processData()

yReg_lr = linearRegression(X_train, X_test, yReg_train, 0)
lrSc = r2_score(yReg_test, yReg_lr)
X_linReg = X_test

yReg_nr = kNNRegression(X_train, X_test, yReg_train, 0)
nrSc = r2_score(yReg_test, yReg_nr)
X_knnReg = X_test

analyseData()

evaluateClassification(yClass_test, yClass_lr, 'Logistic Regression', displayDetailedView = 1)
plotDecisionBoundry(X_test, yClass_test, yClass_lr, 'Logistic Regression')

evaluateClassification(yClass_test, yClass_rf, 'Random Forest', displayDetailedView = 1)
plotDecisionBoundry(X_test, yClass_test, yClass_rf, 'Random Forest')

evaluateClassification(yClass_test, yClass_knn, 'K-nearest neighbors', displayDetailedView = 1)
plotDecisionBoundry(X_test, yClass_test, yClass_knn, 'K-nearest neighbors')

evaluateRegression(yReg_test, yReg_lr, 'Linear Regression')
evaluateRegression(yReg_test, yReg_nr, 'KNN Regression')

X, yClass, yReg = processData(trainTestSplit = 2)
classes = ['German Shepherd', 'Daschhund', 'Samoyed', 'Siberian Husky']

y_kmeans = kmeans(X, init='random')
evaluateClustering(X, yClass.values.ravel(), y_kmeans, 'K-means Clustering', classes)

y_dbscan = dbscan(X)
evaluateClustering(X, yClass.values.ravel(), y_dbscan, 'DBSCAN Clustering', classes)

y_agglomerative = agglomerative(X)
evaluateClustering(X, yClass.values.ravel(), y_agglomerative, 'Agglomerative Clustering', classes)

vizualizeData(X_lr, yClass_lr.ravel(),
              X_rf, yClass_rf.ravel(), X_knn,
              yClass_knn.ravel(),
    										X_linReg, yReg_lr.ravel(),
              X_knnReg, yReg_nr.ravel(),
    										y_kmeans, y_dbscan, y_agglomerative)

print('\nFINAL VERDICT:')
print('\nAccuracy(lr, rf, knn): ', lrAc, rfAc, knnAc)
print('\nScore(lr, nr): ', lrSc, nrSc)