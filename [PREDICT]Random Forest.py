# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 01:12:17 2020

@author: Osman
"""

#1. kutuphaneler(libraries)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme[data uploading]
veriler = pd.read_csv('maaslar.csv')

#data frame dilimleme (slice)
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

#NumPY dizi (array) dönüşümü
X = x.values
Y = y.values


#linear regression
#doğrusal model oluşturma
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#polynomial regression
#doğrusal olmayan (nonlinear model) oluşturma
#2. dereceden polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

# 4. dereceden polinom
poly_reg3 = PolynomialFeatures(degree = 4)
x_poly3 = poly_reg3.fit_transform(X)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y)

# Gorsellestirme(visulation)
plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X), color = 'blue')
plt.show()

plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)), color = 'blue')
plt.show()

#tahminler(predicts)
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[11]])))
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))

#verilerin ölçeklenmesi
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)

from sklearn.svm import SVR
svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli, color = 'yellow')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli), color = 'cyan')
plt.show()
print("RBF(Radial Base Function) ile yapılan Support Vector Regression")
print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))


#DECISION TREE 
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(X,Y)
Z = X + 0.5
K = X - 0.4 #0.5 olmama sebei bir eksik sayıya tamlama yapması
#K ve Z nin amacı ondalıklı sayıların nasıl tamsayıya yuvarlanarak aynı şeklin
#nasıl elde edildiğini göstermek

plt.scatter(X,Y, color = 'blue')
plt.plot(x,dt_reg.predict(X), color = 'cyan')
#oluşan chart öncekine çok benzer çünkü yakın değerler tamsayıya yuvarlanıyor.
plt.show()
plt.scatter(X,Y, color = 'blue')
plt.plot(x,dt_reg.predict(Z), color = 'yellow')
plt.plot(x,dt_reg.predict(K), color = 'orange')
plt.show()
print(dt_reg.predict([[11]]))
print(dt_reg.predict([[6.6]]))

#RASSAL FOREST
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10,random_state = 0)
rf_reg.fit(X,Y)
print("Buradan sonrası Rassal Forest ile çoklu ağaç gibi düşünülebilir.")
print(rf_reg.predict([[6.6]]))

plt.scatter(X,Y, color = 'red')
plt.plot(x,rf_reg.predict(X),color='blue')
plt.plot(x,rf_reg.predict(Z), color = 'green')
plt.plot(x,rf_reg.predict(K), color= 'yellow')







    

