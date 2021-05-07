# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 23:19:01 2020

@author: Osman
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

salaries = pd.read_csv('maaslar.csv')

x = salaries.iloc[:,1:2]
y = salaries.iloc[:,2:]
#x : eğitim durumu(educational state), y : maaslar(salaries) 
X = x.values
Y = y.values
#for easy operating-using(kullanım kolaylığı için)

#Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

plt.scatter(X,Y,color = 'red')
plt.plot(x,lin_reg.predict(X), color = 'green')
plt.show()

#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color = 'purple')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'yellow')
plt.show()

poly_reg = PolynomialFeatures(degree = 4)
#we are doing like Limit(which is in Calculus) so we can reach predict closer 
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly,y)
plt.scatter(X,Y, color = 'orange')
plt.plot(X,lin_reg3.predict(poly_reg.fit_transform(X)), color = 'brown')
plt.show()

poly_reg = PolynomialFeatures(degree = 10)
#we are doing like Limit(which is in Calculus) so we can reach predict closer 
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly,y)
plt.scatter(X,Y, color = 'orange')
plt.plot(X,lin_reg3.predict(poly_reg.fit_transform(X)), color = 'brown')
plt.show()

#Let's check out the regression's predict
#Linear Regression
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

#Polynomial Regression
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))













