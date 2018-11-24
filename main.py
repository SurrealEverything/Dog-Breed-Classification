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

#loading pre-processed data
X_train, X_test, yClass_train, yClass_test, yReg_train, yReg_test = processData()

yClass_lr = logisticRegression(X_train, X_test, yClass_train)
lrAc = accuracy_score(yClass_test, yClass_lr)

#reloading data pre-processed with different parameters 
X_train, X_test, yClass_train, yClass_test, yReg_train, yReg_test = processData(normalization = 'mms')

yClass_rf = randomForest(X_train, X_test, yClass_train)
rfAc = accuracy_score(yClass_test, yClass_rf)

yClass_knn = kNN(X_train, X_test, yClass_train)
knnAc = accuracy_score(yClass_test, yClass_knn)
X_train, X_test, yClass_train, yClass_test, yReg_train, yReg_test = processData()

yReg_lr = linearRegression(X_train, X_test, yReg_train, 0)
lrSc = r2_score(yReg_test, yReg_lr)

yReg_nr = kNNRegression(X_train, X_test, yReg_train, 0)
nrSc = r2_score(yReg_test, yReg_nr)

analyseData()

evaluateClassification(yClass_test, yClass_lr, 'Logistic Regression', displayDetailedView = 1)
plotDecisionBoundry(X_test, yClass_test, yClass_lr, 'Logistic Regression')

evaluateClassification(yClass_test, yClass_rf, 'Random Forest', displayDetailedView = 1)
plotDecisionBoundry(X_test, yClass_test, yClass_rf, 'Random Forest')

evaluateClassification(yClass_test, yClass_knn, 'K-nearest neighbors', displayDetailedView = 1)
plotDecisionBoundry(X_test, yClass_test, yClass_knn, 'K-nearest neighbors')

evaluateRegression(yReg_test, yReg_lr, 'Linear Regression')
evaluateRegression(yReg_test, yReg_lr, 'KNN Regression')

print('\nFINAL VERDICT:')
print('\nAccuracy(lr, rf, knn): ', lrAc, rfAc, knnAc)
print('\nScore(lr, nr): ', lrSc, nrSc)


