# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 23:56:57 2020

@author: Osman
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer


#2.1. Veri Yukleme
veriler = pd.read_csv('veriler.csv')
print(veriler)
#pd.read_csv("veriler.csv")


imputer = Imputer(missing_values='NaN',strategy= 'mean',axis = 0)
Yas = veriler.iloc[:,1:4].values
print(Yas)
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)


#encoder:  Kategorik -> Numeric

#2. Veri Onisleme
ulke = veriler.iloc[:,0:1].values
print(ulke)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])
print(ulke)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)

c = veriler.iloc[:,-1:].values
print(c)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
c[:,0] = le.fit_transform(c[:,0])
print(ulke)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
c=ohe.fit_transform(c).toarray()
print(c)

#numpy dizileri dataframe donusumu
sonuc = pd.DataFrame(data = ulke, index = range(22), columns=['fr','tr','us'] )
print(sonuc)

sonuc2 =pd.DataFrame(data = Yas, index = range(22), columns = ['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = c[:,:1] , index=range(22), columns=['cinsiyet'])
print(sonuc3)

#dataframe birlestirme islemi
s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2= pd.concat([s,sonuc3],axis=1)
print(s2)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_prediction = regressor.predict(x_test)

height = s2.iloc[:,3:4].values
print(height)
left_side = s2.iloc[:,:3]
right_side = s2.iloc[:,4:]

data = pd.concat([left_side,right_side],axis=1)

x_train2, x_test2 , y_train2, y_test2 = train_test_split(data,height,test_size=0.33,random_state=0)


r2 = LinearRegression()
r2.fit(x_train2,y_train2)

y_prediction2 = r2.predict(x_test2)

import statsmodels.api as sm

X = np.append(arr = np.ones((22,1)).astype(int), values=data, axis=1)
X_list = data.iloc[:,[0,1,2,3,4,5]].values
reg_ols = sm.OLS(endog = height, exog = X_list)
reg = reg_ols.fit()
print(reg.summary())

#5. sütunda p value yüksek çıktığı için çıkabilir
X_list = data.iloc[:,[0,1,2,3,5]].values
reg_ols = sm.OLS(endog = height, exog = X_list)
reg = reg_ols.fit()
print(reg.summary())

X_list = data.iloc[:,[0,1,2,3]].values
reg_ols = sm.OLS(endog = height, exog = X_list)
reg = reg_ols.fit()
print(reg.summary())

    
    

