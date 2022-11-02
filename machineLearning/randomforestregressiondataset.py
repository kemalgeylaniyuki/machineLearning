# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 19:00:25 2022

@author: kemal
"""

# random+forest+regression+dataset

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("random+forest+regression+dataset.csv", sep = ";", header = None)

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

#%% fit

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 100, random_state = 42) # (n_estimators = 100) -> 100 adet tree
rf.fit(x,y)

x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)

#%% plot

y_head = rf.predict(x_)

plt.scatter(x, y, color = "red")
plt.plot(x_, y_head, color = "green")
plt.xlabel("Trubin")
plt.ylabel("Fiyat")
plt.show()

#%% R-Square with Random Forest

from sklearn.metrics import r2_score

y_head = rf.predict(x)
print("r_score : ", r2_score(y,y_head))

