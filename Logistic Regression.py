# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 10:41:46 2020

@author: Osman
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

datas = pd.read_csv("veriler.csv")
print(datas)

x = datas.iloc[:,1:4].values # bağımsız değişken(boy,kilo,yaş)
y = datas.iloc[:,4:].values # bağımlı değişken(cinsiyet)

print(y)

#let's split the data for testing and training 
#verileri eğitim ve test için bölelim
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size = 0.33,random_state=0)

#scaling data
#verilerin ölçeklenmesi 
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train= sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)

















