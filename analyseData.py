#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 22:45:55 2018

@author: gabriel
"""
import matplotlib.pyplot as plt
import pandas as pd
import gender_guesser.detector as gender
from get_binary_gender import get_binary_gender
import numpy as np
from sklearn import preprocessing
from collections import Counter
import seaborn as sns

def processData():
	df = pd.read_csv('Dumitrescu_Gabriel_Horia_train.csv')
	
	d = gender.Detector()
	df['Owner Sex'] =  np.vectorize(get_binary_gender)(d, df['Owner Name'])
			
	df.drop(['Owner Name'], axis=1, inplace=True)
	
	df.fillna(df.mean(), inplace=True)

	return df

def printStats(df):
	stats = df.describe(include='all')
	print(stats)

def plotDistribution(df, col):
	counts = Counter(df[col])
	df1 = pd.DataFrame.from_dict(counts, orient='index')
	ax = df1.plot(kind='bar', title='Distribution of ' + col)
	ax.set_xlabel(col)
	ax.set_ylabel('Count')

def plotDistributions(df):
	for i in ['Breed Name', 'Energy level', 'Attention Needs', 'Coat Lenght', 'Sex', 'Owner Sex']:
		plotDistribution(df, i)

def corr(x, y, **kwargs):
    
    # Calculate the value
    coef = np.corrcoef(x, y)[0][1]
    # Make the label
    label = r'$\rho$ = ' + str(round(coef, 2))
    
    # Add the label to the plot
    ax = plt.gca()
    ax.annotate(label, xy = (0.2, 0.95), size = 20, xycoords = ax.transAxes)
    
# Create a pair plot colored by Breed Name with a density plot of the # diagonal and format the scatter plots.
def plotPair(df):
	sns.pairplot(df, vars = ['Weight(g)', 'Height(cm)', 'Longevity(yrs)'], hue = 'Breed Name', diag_kind = 'kde',
	             plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'}, size = 4)

def plotGrid(df):
	grid = sns.PairGrid(data = df, vars = ['Weight(g)', 'Height(cm)', 'Longevity(yrs)'], size = 4)
	#plt.suptitle('Data Distributions and Clusters', size = 28);
	
	# Map a scatter plot to the upper triangle
	grid = grid.map_upper(plt.scatter, color = 'darkred')
	grid = grid.map_upper(corr)
	# Map a histogram to the diagonal
	grid = grid.map_diag(plt.hist, bins = 10, color = 'darkred', edgecolor = 'k')
	# Map a density plot to the lower triangle
	grid = grid.map_lower(sns.kdeplot, cmap = 'Reds')

def encodeLabels(df):
	le = preprocessing.LabelEncoder()
	for i in [0]+list(range(4, 8)):
		npyClass = le.fit_transform(df.iloc[:, i])
		df.iloc[:, i] = npyClass

#Function plots a graphical correlation matrix for each pair of columns in the dataframe
def plot_corr(df,size=13):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    print (corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title('Correlation Matrix(yellow=big, purple=small)')
    
def analyseData():
	df = processData()
	printStats(df)
	plotDistributions(df)
	plotPair(df)
	#plotGrid(df)
	encodeLabels(df)
	plot_corr(df)
	plt.show()