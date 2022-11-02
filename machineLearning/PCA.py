# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 20:21:43 2022

@author: kemal
"""

from sklearn.datasets import load_iris
import pandas as pd

#%% 

iris = load_iris()

data = iris.data
feature_names = iris.feature_names
y = iris.target

df = pd.DataFrame(data, columns = feature_names)
df["sinif"] = y
x = data

#%%

from sklearn.decomposition import PCA

pca = PCA(n_components = 2, whiten = True) # whiten : for normalization
pca.fit(x)
x_pca = pca.transform(x)

print("Variance ratio : ", pca.explained_variance_ratio_)
print("Sum : ", sum(pca.explained_variance_ratio_))

#%% Visualize

df["p1"] = x_pca[:,0]
df["p2"] = x_pca[:,1]

color = ["red", "green", "blue"]

import matplotlib.pyplot as plt

for i in range(3):  
    plt.scatter(df.p1[df.sinif == i], df.p2[df.sinif == i], color = color[i], label = iris.target_names[i])
    
plt.legend()
plt.xlabel("P1")
plt.ylabel("P2")
plt.show()

