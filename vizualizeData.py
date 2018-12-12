#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 02:31:31 2018

@author: gabriel
"""
from mpl_toolkits import mplot3d
from ipywidgets import interact, fixed
import matplotlib.pyplot as plt
from dataPreprocessing import processData
from sklearn.manifold.t_sne import TSNE
from sklearn.decomposition import PCA


def isInteractive():
    import __main__ as main
    return not hasattr(main, '__file__')


def plot_3D(elev, azim, X, y, title):
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=y, s=50)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title(title)


def plotInteractive(X, y, title):
    interact(plot_3D, elev=(-90, 90), azim=(-180, 180),
             X=fixed(X), y=fixed(y), title=fixed(title))


def plotStationary(X, y, title):
    ax = plt.axes(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=y)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title(title)
    plt.show()


def vizualizeData(X_lr, y_lr, X_rf, y_rf, X_knn, y_knn,
																		X_linReg, y_linReg, X_knnReg, y_knnReg,
																		y_kmeans, y_dbscan, y_agglomerative):

    interactive = isInteractive()

			 # X_PCA and X_TSNE
    X, yClass, yReg = processData(trainTestSplit = 2)
    yClass = yClass.values.ravel()
    yReg = yReg.values.ravel()
    X_PCA = PCA(n_components=3).fit_transform(X)
    X_TSNE = TSNE(n_components=3).fit_transform(X)

    X_lr_PCA = PCA(n_components=3).fit_transform(X_lr)
    X_lr_TSNE = TSNE(n_components=3).fit_transform(X_lr)

    X_rf_PCA = PCA(n_components=3).fit_transform(X_rf)
    X_rf_TSNE = TSNE(n_components=3).fit_transform(X_rf)

    X_knn_PCA = PCA(n_components=3).fit_transform(X_knn)
    X_knn_TSNE = TSNE(n_components=3).fit_transform(X_knn)

    X_linReg_PCA = PCA(n_components=3).fit_transform(X_linReg)
    X_linReg_TSNE = TSNE(n_components=3).fit_transform(X_linReg)

    X_knnReg_PCA = PCA(n_components=3).fit_transform(X_knnReg)
    X_knnReg_TSNE = TSNE(n_components=3).fit_transform(X_knnReg)

    if interactive:
        plotInteractive(X_PCA, yClass, 'PCA Classification')
        plotInteractive(X_TSNE, yClass, 'TSNE Classification')

        plotInteractive(X_PCA, yReg, 'PCA Regression')
        plotInteractive(X_TSNE, yReg, 'TSNE Regression')

        plotInteractive(X_lr_PCA, y_lr, 'Logistic Regression PCA')
        plotInteractive(X_lr_TSNE, y_lr, 'Logistic Regression TSNE')

        plotInteractive(X_rf_PCA, y_rf, 'Random Forests PCA')
        plotInteractive(X_rf_TSNE, y_rf, 'Random Forests TSNE')

        plotInteractive(X_knn_PCA, y_knn, 'K-nearest neighbors PCA')
        plotInteractive(X_knn_TSNE, y_knn, 'K-nearest neighbors TSNE')

        plotInteractive(X_linReg_PCA, y_linReg, 'Linear Regression PCA')
        plotInteractive(X_linReg_TSNE, y_linReg, 'Linear Regression TSNE')

        plotInteractive(X_knnReg_PCA, y_knnReg, 'KNN Regression PCA')
        plotInteractive(X_knnReg_TSNE, y_knnReg, 'KNN Regression TSNE')

        plotInteractive(X_PCA, y_kmeans, 'K-means Clustering PCA')
        plotInteractive(X_TSNE, y_kmeans, 'K-means Clustering TSNE')

        plotInteractive(X_PCA, y_dbscan, 'DBSCAN Clustering PCA')
        plotInteractive(X_TSNE, y_dbscan, 'DBSCAN Clustering TSNE')

        plotInteractive(X_PCA, y_agglomerative, 'Agglomerative Clustering PCA')
        plotInteractive(X_TSNE, y_agglomerative,
                        'Agglomerative Clustering TSNE')

    else:
        plotStationary(X_PCA, yClass, 'PCA Classification')
        plotStationary(X_TSNE, yClass, 'TSNE Classification')

        plotStationary(X_PCA, yReg, 'PCA Regression')
        plotStationary(X_TSNE, yReg, 'TSNE Regression')

        plotStationary(X_lr_PCA, y_lr, 'Logistic Regression PCA')
        plotStationary(X_lr_TSNE, y_lr, 'Logistic Regression TSNE')

        plotStationary(X_rf_PCA, y_rf, 'Random Forests PCA')
        plotStationary(X_rf_TSNE, y_rf, 'Random Forests TSNE')

        plotStationary(X_knn_PCA, y_knn, 'K-nearest neighbors PCA')
        plotStationary(X_knn_TSNE, y_knn, 'K-nearest neighbors TSNE')

        plotStationary(X_linReg_PCA, y_linReg, 'Linear Regression PCA')
        plotStationary(X_linReg_TSNE, y_linReg, 'Linear Regression TSNE')

        plotStationary(X_knnReg_PCA, y_knnReg, 'KNN Regression PCA')
        plotStationary(X_knnReg_TSNE, y_knnReg, 'KNN Regression TSNE')

        plotStationary(X_PCA, y_kmeans, 'K-means Clustering PCA')
        plotStationary(X_TSNE, y_kmeans, 'K-means Clustering TSNE')

        plotStationary(X_PCA, y_dbscan, 'DBSCAN Clustering PCA')
        plotStationary(X_TSNE, y_dbscan, 'DBSCAN Clustering TSNE')

        plotStationary(X_PCA, y_agglomerative, 'Agglomerative Clustering PCA')
        plotStationary(X_TSNE, y_agglomerative,
                        'Agglomerative Clustering TSNE')
