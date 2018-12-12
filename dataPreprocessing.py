#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 23:47:29 2018

@author: gabriel
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import gender_guesser.detector as gender
from get_binary_gender import get_binary_gender
from imblearn.over_sampling import SMOTE

def processData(
filename = 'Dumitrescu_Gabriel_Horia_train.csv',
includeOwnerGender=1,
computeStats=0,
dropList = ['Owner Name'],
dropFirstOH = 1,
normalization = 'ss',
trainTestSplit = 0,
randomSeed = 2,
oversample = 0,
addBigHousePet = 1):
	#incarcam datele
	df = pd.read_csv(filename)

	#adaugam featurul: Owner Sex
	if includeOwnerGender:
		d = gender.Detector()
		df['Owner Sex'] =  np.vectorize(get_binary_gender)(d, df['Owner Name'])

	#adaugam clasa de greutate
	if addBigHousePet:
		lowWeight = 14291
		highWeight = 20680
		lowHeight = 37
		ind = df[( df['Weight(g)'] < highWeight) & ( df['Weight(g)'] > lowWeight) & ( df['Height(cm)'] > lowHeight) &
		     (df['Attention Needs'] == 'med')].index.values
		df['Big House Pet'] = np.zeros((1000,))
		df.iloc[ind, df.columns.get_loc('Big House Pet')] = 1

	#stergem coloane
	df.drop(dropList, axis=1, inplace=True)

	#completam campurile goale cu media pe coloana a celorlalte campuri
	df.fillna(df.mean(), inplace=True)

	#separam Breed Name
	yClass = df.iloc[:, 0]
	yClass = yClass.to_frame()
	df.drop(['Breed Name'], axis=1, inplace=True)

	#encodare numerica a labelurilor din format String
	le = preprocessing.LabelEncoder()
	npyClass = le.fit_transform(yClass)
	yClass = pd.DataFrame(npyClass, columns = yClass.columns)

	#encodare Onehot a categoriilor tip String
	df = pd.get_dummies(df, drop_first = dropFirstOH)

	#oversampling
	if oversample:
		npdf, npyClass = SMOTE().fit_sample(df, yClass)
		df = pd.DataFrame(npdf, columns = df.columns)
		yClass = pd.DataFrame(npyClass, columns = yClass.columns)

	#Longevity(yrs)
	yReg = df.iloc[:, 2]
	yReg = yReg.to_frame()
	df.drop(['Longevity(yrs)'], axis=1, inplace=True)

	#restul
	X = df

	#normalizam X cu una dintre urmatoarele tehnici
	if normalization == 'mms':
		scaler = preprocessing.MinMaxScaler()
	elif normalization == 'ss':
		scaler = preprocessing.StandardScaler()
	elif normalization == 'mas':
		scaler = preprocessing.MaxAbsScaler()
	elif normalization == 'qt':
		scaler = preprocessing.QuantileTransformer()
	elif normalization == 'rs':
		scaler = preprocessing.RobustScaler()
	elif normalization == 'n':
		scaler = preprocessing.Normalizer()

	if normalization != 'None':
		npX = scaler.fit_transform(X)
		X = pd.DataFrame(npX, columns = X.columns)

	#statistici date
	if computeStats:
		statsX = X.describe(include='all')
		statsyClass = yClass.describe(include='all')
		statsyReg = yReg.describe(include='all')
		print(statsX, statsyClass, statsyReg)

	if trainTestSplit==1:
		#impartim datele in multimi de test si de training
		X_train, X_test, yClass_train, yClass_test = train_test_split(X, yClass, random_state=randomSeed)
		X_train, X_test, yReg_train, yReg_test = train_test_split(X, yReg, random_state=randomSeed)
		return X_train, X_test, yClass_train, yClass_test, yReg_train, yReg_test

	elif trainTestSplit==0:
		#impartim datele in multimi de test si de training reprezentative
		sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=randomSeed)
		for train_index, test_index in sss.split(X, yClass):
			print("TRAIN:", train_index, "TEST:", test_index)
			X_train, X_test = X.iloc[train_index], X.iloc[test_index]
			yClass_train, yClass_test = yClass.iloc[train_index], yClass.iloc[test_index]
			yReg_train, yReg_test = yReg.iloc[train_index], yReg.iloc[test_index]

			return X_train, X_test, yClass_train, yClass_test, yReg_train, yReg_test

	else:
		return X, yClass, yReg

#X_train, X_test, yClass_train, yClass_test, yReg_train, yReg_test = processData()