# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 11:59:39 2020

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

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = "entropy") #default olarak gini verilir.

dtc.fit(X_train,y_train)

y_pred = dtc.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print("\nDecision Tree(entropy) ile sınıflandırılırsa;")
print(cm)

dtc2 = DecisionTreeClassifier() #☻default olarak gini atanır.

dtc2.fit(X_train,y_train)

y_pred2 = dtc2.predict(X_test)

cm2 = confusion_matrix(y_test,y_pred2)
print("\nDecision Tree(gini) ile yapılırsa;")
print(cm2)
















