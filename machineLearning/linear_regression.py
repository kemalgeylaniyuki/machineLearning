# -*- coding: utf-8 -*-
"""
Spyder Editor
gg
This is a temporarygg script file.
"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("linear_regression_dataset.csv", sep = ";")
plt.scatter(df.deneyim, df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()

#%% 

# Linear Regression

from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()

x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)

#%%

# Predict

import numpy as np

b0 = linear_reg.predict([[0]])

b0_ = linear_reg.intercept_
print("b0 : ",b0_)          # y ekseni kesen nokta : intercept

b1 = linear_reg.coef_
print("b1 : ", b1)  # eğim slope

# maas = y = b0 + b0*x(deneyim)
# maas = 1663 + 1138*x(deneyim)

maas_yeni = 1663 + 1138*11

print(linear_reg.predict([[11]]))

# Visualize Line

array = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1) # deneyim

plt.scatter(x,y)
plt.show()

y_head = linear_reg.predict(array) # maas

plt.plot(array, y_head, color = "red")

