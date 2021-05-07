# -*- coding: utf-8 -*-
"""
Created on Fri May 22 00:39:31 2020

@author: Osman
"""

import pandas as pd

datas = pd.read_csv('satislar.csv')

months = datas[['Aylar']]
print(months)

sells = datas[['Satislar']]
print(sells)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(months,sells,test_size=0.33,random_state=0)

#building(structure) of model
#model inşası

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)

estimate = lr.predict(x_test)
# predict = estimate = tahmin






