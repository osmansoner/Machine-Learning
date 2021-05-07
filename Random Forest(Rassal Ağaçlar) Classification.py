# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 12:42:33 2020

@author: Osman
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

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

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=10,criterion="entropy",random_state=0)#default olarak 100 ve gini verilir..
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print("\nRandom Forest ile yapılırsa;")
print(cm)


rfc2 = RandomForestClassifier(n_estimators=10000,criterion="entropy",random_state=0)
#estimator i çok arttırmak bir şey ifade etmez.
rfc2.fit(X_train,y_train)

y_pred2 = rfc2.predict(X_test)

cm2 = confusion_matrix(y_test,y_pred2)
print("\nRandom Forest(estimator yüksek) ile yapılırsa;")
print(cm2)


rfc3 = RandomForestClassifier(n_estimators=10,criterion="gini",random_state=0)
rfc3.fit(X_train,y_train)

y_pred3 = rfc3.predict(X_test)

from sklearn.metrics import confusion_matrix
cm3 = confusion_matrix(y_test,y_pred3)
print("\nRandom Forest(gini) ile yapılırsa;")
print(cm3)










