# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 11:33:34 2020

@author: Osman
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datas = pd.read_csv("veriler.csv")

print(datas)

x = datas.iloc[:,1:4].values
y = datas.iloc[:,4:].values

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors=5,metric = "minkowski")
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

#confusion matrix i yazalım(karmaşıklık matrisi)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print("KNN n_neighbors = 5 ile yapılan(minkowski)")
print(cm) 

knn2 = KNeighborsClassifier(n_neighbors=1,metric = "minkowski")
knn2.fit(X_train,y_train)

y_pred2 = knn2.predict(X_test)

cm2 = confusion_matrix(y_test,y_pred2)
print("KNN sadece 1 komşuluk ile yapılırsa(minkowski)")
print(cm2)

knn3 = KNeighborsClassifier(n_neighbors=1,metric="euclidean")
knn3.fit(X_train,y_train)

y_pred3 = knn3.predict(X_test)

cm3 = confusion_matrix(y_test,y_pred3)
print("\nKNN 1 komşuluk euclidean ile yapılırsa")
print(cm3)

knn4 = KNeighborsClassifier(n_neighbors=5,metric="euclidean")
knn4.fit(X_train,y_train)

y_pred4 = knn4.predict(X_test)

cm4 = confusion_matrix(y_test,y_pred4)
print("\nKNN 1 komşuluk euclidean ile yapılırsa")
print(cm4)










