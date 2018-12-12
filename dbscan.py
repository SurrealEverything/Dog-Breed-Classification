#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 04:40:29 2018

@author: gabriel
"""
from sklearn.cluster import DBSCAN


def dbscan(X, eps=2, min_samples=80):

    model = DBSCAN(eps=eps, min_samples=min_samples)

    model.fit(X)

    y = model.labels_

    return y
