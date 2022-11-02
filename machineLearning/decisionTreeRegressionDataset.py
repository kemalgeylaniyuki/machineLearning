# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 20:42:06 2022

@author: kemal
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("decision+tree+regression+dataset.csv", sep = ";", header = None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

#%% Decision Tree

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()

tree_reg.fit(x,y)

x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = tree_reg.predict(x_)

#%% Visualize

plt.scatter(x,y,color="red")
plt.plot(x_,y_head, color = "green")
plt.xlabel("Trubin")
plt.ylabel("Ucret")
plt.show()

