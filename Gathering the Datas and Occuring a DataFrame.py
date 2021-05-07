# -*- coding: utf-8 -*-
"""
Created on Fri May 15 01:37:34 2020

@author: Osman
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer

datas = pd.read_csv("veriler.csv")

country = datas.iloc[:,0:1].values
print(country)

label_enc = LabelEncoder()

country[:,0] = label_enc.fit_transform(country[:,0])
print(country) #1 - 2 - 0 gibi ülkelerin nominal(isim) yerine etiket verdik.

ohe = OneHotEncoder(categorical_features="all")
country = ohe.fit_transform(country).toarray()
print(country)

#to indicate how looks a list 
print(list(range(22))) #liste şeklinde yazmayı göstermek için

#we gave the names of countries to the columns 
result = pd.DataFrame(data = country, index = range(22),columns = ['fr','tr','us'] )
print(result)

imputer = SimpleImputer(missing_values=np.nan, strategy = "mean")
Age = datas.iloc[:,1:4].values
print(Age)
imputer = imputer.fit(Age[:,1:4])
Age[:,1:4] = imputer.transform(Age[:,1:4])
print(Age) 



result2 = pd.DataFrame(data = Age, index = range(22),columns = ['boy','kilo','cinsiyet'])
print(result2)

#if we wanna get the latest column then we have to take with -1 
gender = datas.iloc[:,-1].values
print(gender)

#we're writing the converted a dataframe from list of gender 
result3 = pd.DataFrame(data = gender, index = range(22), columns= ['cinsiyet'])
print(result3)

#it would write not gathered it would just directly beneathe the dataframe 
r = pd.concat([result,result2])
print(r)

#if  the axis = 1 it would write with gathering
r = pd.concat([result,result2],axis=1) # axis = 1 olursa satır bazından birleştirir.
print(r)

r2 = pd.concat([r,result3],axis=1) #dataframe tamamlanmış oldu tek farkı ülkeler nomimal değerde değil.
print(r2)
#we completed the DataFrame but there are differents with the begining one.
#Differents are the countries names not nominal and there is no NaN values anymore.   



