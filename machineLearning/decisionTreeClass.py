# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 15:06:55 2022

@author: kemal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%

data = pd.read_csv("data.csv")
data.head()
data.drop(["id","Unnamed: 32"], axis = 1, inplace = True)

data.head()

M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]

plt.scatter(M.radius_mean, M.texture_mean, color = "red", label = "Bad", alpha = 0.3)
plt.scatter(B.radius_mean, B.texture_mean, color = "green", label = "Good", alpha = 0.3)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()

data.diagnosis = [1 if i == "M" else 0 for i in data.diagnosis]
y = data.diagnosis.values
x_ = data.drop(["diagnosis"], axis = 1)

#%% # Normalizations
x = (x_ - np.min(x_))/(np.max(x_)-np.min(x_))

#%% train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15, random_state = 42)    

#%% 
from sklearn.tree import DecisionTreeClassifier
d_tree = DecisionTreeClassifier()
d_tree.fit(x_train,y_train)

print("Score : ", d_tree.score(x_test,y_test))
