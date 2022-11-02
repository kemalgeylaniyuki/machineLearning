# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 11:11:43 2018

@author: user
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("polynomial+regression.csv", sep = ";")

y = df.araba_max_hiz.values.reshape(-1,1)
x = df.araba_fiyat.values.reshape(-1,1)

plt.scatter(x, y)
plt.ylabel("araba_max_hiz")
plt.xlabel("araba_fiyat")
plt.show()

# Linear regersion : y = b0 + b1*x
# Multiple linear regression : y = b0 + b1*x1 + b2x2

#%%

# Linear regression

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x,y)

#%%

# Predict

y_head = lr.predict(x)

plt.plot(x, y_head, color = "red", label = "Linear")
plt.legend()
plt.show()

#%%

# polinomial regression : y = b0 b1*x1 + b2*x2^2 ....

from sklearn.preprocessing import PolynomialFeatures
pol_regression = PolynomialFeatures(degree = 4)

x_polinomial = pol_regression.fit_transform(x)

#%%

# fit()

lr2 = LinearRegression()
lr2.fit(x_polinomial,y)

#%%

# Predict

y_head2 = lr2.predict(x_polinomial)

plt.plot(x,y_head2, color = "yellow", label = "Polynomial")
plt.legend()
plt.show()
















