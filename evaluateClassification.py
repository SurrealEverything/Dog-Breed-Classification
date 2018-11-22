#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 04:42:47 2018

@author: gabriel
"""
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
def evaluateClassification(y_true, y_pred, model, displayDetailedView = False, plotConfusion = True):
	
	print('\n', model.upper())
	
	if displayDetailedView:
		acc = accuracy_score(y_true, y_pred)
		print('Accuracy: ', acc)
		
		prMac = precision_score(y_true, y_pred, average='macro')  
		prMic = precision_score(y_true, y_pred, average='micro')  
		prWgh = precision_score(y_true, y_pred, average='weighted')
		print('\nMacro Precision: ', prMac)
		print('Micro Precision: ', prMic)
		print('Weighted Precision: ', prWgh)
		
		rcMac = recall_score(y_true, y_pred, average='macro')  
		rcMic = recall_score(y_true, y_pred, average='micro')  
		rcWgh = recall_score(y_true, y_pred, average='weighted')
		print('\nMacro Recall: ', rcMac)
		print('Micro Recall: ', rcMic)
		print('Weighted Recall: ', rcWgh)
		
		f1Mac = f1_score(y_true, y_pred, average='macro')  
		f1Mic = f1_score(y_true, y_pred, average='micro')  
		f1Wgh = f1_score(y_true, y_pred, average='weighted')
		print('\nMacro F1: ', f1Mac)
		print('Micro F1: ', f1Mic)
		print('Weighted F1: ', f1Wgh)
	
	
	target_names = ['German Shepherd', 'Daschhund', 'Samoyed', 'Siberian Husky']
	print(classification_report(y_true, y_pred, target_names=target_names))
	
	if plotConfusion:
		# Compute confusion matrix
		cnf_matrix = confusion_matrix(y_true, y_pred)
		#np.set_printoptions(precision=2)
		
		# Plot non-normalized confusion matrix
		plt.figure()
		plot_confusion_matrix(cnf_matrix, classes=target_names,
		                      title = (model + ' Confusion Matrix'))
		plt.show()
		