# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 20:01:49 2022

@author: kemal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Create Dataset

#Class 1
x1 = np.random.normal(25,5,100) # 1000 tane random sayı, ortalaması 25
y1 = np.random.normal(25,5,100)

#Class 2
x2 = np.random.normal(55,5,100)
y2 = np.random.normal(60,5,100)

#Class 3
x3 = np.random.normal(55,5,100)
y3 = np.random.normal(15,5,100)

x = np.concatenate((x1,x2,x3), axis = 0) # x1 x2 ve x3 ü dikey satırda birleştir.
y = np.concatenate((y1,y2,y3), axis = 0)

dictionary = {"x" : x, "y" : y}

data = pd.DataFrame(dictionary)

plt.scatter(x1,y1, color = "black")
plt.scatter(x2,y2, color = "black")
plt.scatter(x3,y3, color = "black")
plt.show()

#%% Dendrogram

from scipy.cluster.hierarchy import linkage, dendrogram

merg = linkage(data, method = "ward")
dendrogram(merg,leaf_rotation=90)

plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()

#%% HC

from sklearn.cluster import AgglomerativeClustering

hierartical_cluster = AgglomerativeClustering(n_clusters = 3, affinity = "euclidean", linkage = "ward")
cluster = hierartical_cluster.fit_predict(data)

data["label"] = cluster

plt.scatter(data.x[data.label == 0], data.y[data.label == 0], color = "red")
plt.scatter(data.x[data.label == 1], data.y[data.label == 1], color = "green")
plt.scatter(data.x[data.label == 2], data.y[data.label == 2], color = "blue")
plt.show()
































