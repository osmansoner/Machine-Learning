# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 14:17:54 2020

@author: Osman
"""
import pandas as pd


clients = pd.read_csv("Churn_Modelling.csv")

country = clients.iloc[:,4:5].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

country[:,0] = le.fit_transform(country[:,0])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categories = 'auto')
country = ohe.fit_transform(country).toarray()

gender = clients.iloc[:,5:6].values
gender[:,0] = le.fit_transform(gender[:,0])

credit = clients.iloc[:,3:4].values
rest = clients.iloc[:,6:13].values

country_column = pd.DataFrame(data = country, index = range(10000), columns = ["fr","gr","sp"])
rest_column = pd.DataFrame(data = rest, index = range(10000), columns =["Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSlary"])
gender_col = pd.DataFrame(data = gender, index = range(10000), columns = ["gender"])
credit_col = pd.DataFrame(data = credit, index = range(10000), columns = ["CreditScore"])
result1 = pd.concat([credit_col,country_column], axis =1)
result2 = pd.concat([result1,gender_col], axis =1)
result = pd.concat([result2,rest_column], axis =1)
X = result.iloc[:,1:13].values
Y = clients.iloc[:,13].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(6, init = 'uniform', activation='relu', input_dim=11))
#classifier.add(Dense(6, init = 'uniform', activation = 'relu') #neden çalışmadı??















