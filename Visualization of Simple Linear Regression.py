# -*- coding: utf-8 -*-
"""
Created on Fri May 22 01:02:21 2020

@author: Osman
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
datas = pd.read_csv('satislar.csv')
print(datas)

months = datas[['Aylar']]
print(months)

sells = datas[['Satislar']]
print(sells)

x_train, x_test, y_train, y_test = train_test_split(months,sells,test_size=0.33,random_state=0)

lr = LinearRegression()
lr.fit(x_train,y_train)
estimate = lr.predict(x_test)
'''
plt.plot(x_train,y_train)
'''
x_train = x_train.sort_index()
y_train = y_train.sort_index()
#if you don't sort it'll look like too complicated.it's meanless. 
#bu sıralama yapılmazsa çok saçma görüntü oluşur.doğrudur ama 
#sırasız olduğu için anlamsız karışık görünür.

plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))

plt.title("Aylara Göre Satışlar")
plt.xlabel("Aylar")
plt.ylabel("Satış Miktarları")