# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 19:41:43 2022

@author: kemal
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

#%% 

iris = load_iris()

x = iris.data
y = iris.target

#%% Normalization

x = (x - np.min(x))/(np.max(x)-np.min(x))

#%% train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

#%% KNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)


#%% K Fold CV,  K = 10

from sklearn.model_selection import cross_val_score

accuries = cross_val_score(estimator = knn, X = x_train, y = y_train, cv = 10)

print("Avearge Accuracy : ", np.mean(accuries))
print("Avearge std : ", np.std(accuries))

#%%

knn.fit(x_train,y_train)

print("Test Accuracy : ", knn.score(x_test, y_test))

#%% Grid search cross validation

from sklearn.model_selection import GridSearchCV

grid = {"n_neighbors": np.arange(1,50)}

knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, grid, cv = 10) 
knn_cv.fit(x,y)

#%% Print hyperparamter KNN deki K değeri

print("Tuned hyperparameter K : ", knn_cv.best_params_)
print("Tuned parametreye göre en iyi accuracy : ", knn_cv.best_score_)














































