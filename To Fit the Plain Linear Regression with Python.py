# -*- coding: utf-8 -*-
"""
Created on Thu May 21 03:26:23 2020

@author: Osman
"""

import pandas as pd

datas = pd.read_csv('satislar.csv')

months = datas[['Aylar']]
print(months)

sells = datas[['Satislar']]
print(sells)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(months,sells,test_size = 0.33,random_state=0)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_old = sc.fit_transform(x_train)
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

#model inşası
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train)
