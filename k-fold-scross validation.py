# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 17:15:37 2020

@author: Osman
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv("veriler.csv")

x = dataset.iloc[:,1:4].values
y = dataset.iloc[:,4:].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.svm import SVC
classifier = SVC(kernel="rbf",random_state=0)

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print("SVC ile(rbf)")
print(cm)


#k-katlamalı çapraz doğrulama
from sklearn.model_selection import cross_val_score
"""
1. estimator : classifier
2. X
3. Y
4. cv : kaç katlamalı
"""

success = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 4)
print("Accuracy(Başarı) Ortalaması")
print(success.mean())
print("Standart Sapması")
print(success.std())















