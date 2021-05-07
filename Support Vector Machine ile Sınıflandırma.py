# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 12:18:20 2020

@author: Osman
"""
#SVM KERNEL TRICK hakkında da araştırabilirsin.
#Kernel Trick 3 boyutlu bir koni oluşturmayla alakalıydı belki şimdi hatırlarsın..
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

datas = pd.read_csv("veriler.csv")

x = datas.iloc[:,1:4].values
y = datas.iloc[:,4:].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.svm import SVC
svc = SVC(kernel="rbf")

svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print("SVC ile(rbf)")
print(cm)

svc2 = SVC(kernel="linear")

svc2.fit(X_train,y_train)

y_pred2 = svc2.predict(X_test)

cm2 = confusion_matrix(y_test,y_pred2)
print("Svc ile(linear)")
print(cm2)

svc3 = SVC(kernel="poly")

svc3.fit(X_train,y_train)

y_pred3 = svc3.predict(X_test)

cm3 = confusion_matrix(y_test,y_pred3)
print("\nSvc ile(poly)")
print(cm3)

svc4 = SVC(kernel="sigmoid")

svc4.fit(X_train,y_train)

y_pred4 = svc4.predict(X_test)

cm4 = confusion_matrix(y_test,y_pred4)
print("\nSvc ile(sigmoid)")
print(cm4)








