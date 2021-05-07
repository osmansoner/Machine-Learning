# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:32:01 2020

@author: Osman
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

datas = pd.read_csv("veriler.csv")
print(datas)

x = datas.iloc[:,1:4].values
y = datas.iloc[:,4:].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print("\nGaussian Naive Bayes ile tahmin")
print(cm)


"""
from sklearn.naive_bayes import MultinomialNB
mnb =MultinomialNB()

mnb.fit(X_train,y_train)

y_pred2 = mnb.predict(X_test)

cm2 = confusion_matrix(y_test,y_pred2)
print("\nMultinomial Naive Bayes ile tahmin")
print(cm2)

"""


from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()

bnb.fit(X_train,y_train)

y_pred3 = bnb.predict(X_test)

cm3 = confusion_matrix(y_test,y_pred3)
print("\nBernoulli Naive Bayes ile tahmin")
print(cm3)

















