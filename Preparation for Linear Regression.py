# -*- coding: utf-8 -*-
"""
Created on Wed May 20 01:48:44 2020

@author: Osman

Doğrusal Regresyon Hazırlık
"""

import pandas as pd

#2.1. Veri Yukleme
datas = pd.read_csv('satislar.csv')
print(datas)
#pd.read_csv("veriler.csv")


#veri on isleme
months = datas[['Aylar']]
#test
print(months)

#parçalamanın diğer bir yöntemi iloc ile almaktır.
#the other way of seperate ise  like using iloc. 
months2 = datas.iloc[:,:1]
print(months)

sells = datas[['Satislar']]
print(sells)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(months,sells,test_size=0.33, random_state=0)
#parantez içinde ilk önce bağımsız değişken sonra bağımlı değişken gelmelidir.
#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)



'''



    
    

 



