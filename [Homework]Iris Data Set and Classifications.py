# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 10:59:41 2020

@author: Osman
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix


#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_excel('iris.xls')
#pd.read_csv("veriler.csv")
#test
print(veriler)

x = veriler.iloc[:,0:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken
print(y)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#Sınıflandırmalar
#1. Logistic Regression
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)


cm = confusion_matrix(y_test,y_pred)
print("Logistic Regression")
print(cm)


#2. KNN 
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print("KNN")
print(cm)
knn = KNeighborsClassifier(n_neighbors=5,metric='minkowski')
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test) 
cm = confusion_matrix(y_test,y_pred)
print("KNN 5 komşuluk ile")
print(cm)

#3. SVC(SVM Classifier)
from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('SVC')
print(cm)

svc = SVC(kernel='poly')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print("Poly ile SVC")
print(cm)


#4. Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('GNB')
print(cm)

from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
bnb.fit(X_train,y_train)

y_pred = bnb.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print("BernoulliNB")
print(cm)




#5. Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('DTC(entropy)')
print(cm)
dtc = DecisionTreeClassifier() #default old için gini olur.
dtc.fit(X_train,y_train)

y_pred = dtc.predict(X_test)
cm =confusion_matrix(y_test,y_pred)
print("Gini ile Decison Tree")
print(cm)



#6. Random Forest(Rassal Ağaçlar)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('RFC')
print(cm)

rfc = RandomForestClassifier(n_estimators=10,criterion = 'gini')
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print("RFC w/gini")
print(cm)

    
# 7. ROC , TPR, FPR değerleri 

y_proba = rfc.predict_proba(X_test)
print(y_test)
print(y_proba[:,0])

from sklearn import metrics
fpr , tpr , thold = metrics.roc_curve(y_test,y_proba[:,0],pos_label='e')
print(fpr)
print(tpr)
