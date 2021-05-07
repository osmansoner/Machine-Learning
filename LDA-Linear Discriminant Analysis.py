# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 14:48:08 2020

@author: Osman
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
data = pd.read_csv("Wine.csv")

X = data.iloc[:,0:13].values
y = data.iloc[:,13].values

from sklearn.model_selection import train_test_split
X_train, X_test ,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



#Principal Component Analysis(PCA)
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)

X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

#PCA dönüşümünden önce Linear Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train)

#PCA dönüşümünden gelen Linear Rgeression
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(X_train2,y_train)

#Tahminler 
y_pred = classifier.predict(X_test)
y_pred2 = classifier2.predict(X_test2)


from sklearn.metrics import confusion_matrix

#Actual / Pca gerçekleşmeden önce LR
cm = confusion_matrix(y_test, y_pred)
print("Gerçek / PCA gerçekleşmeden önce Linear Regression")
print(cm)

#Actual / PCA gerçekleştikten sonra LR
cm2 = confusion_matrix(y_test, y_pred2)
print("Gerçek / PCA gerçekleştikten sonra Linear Regression")
print(cm2)

#PCA siz ve PCA ile karşılaştırma
cm3= confusion_matrix(y_pred, y_pred2)
print("Comparison w/o PCA  vs  w PCA")
print(cm3)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)

X_train_lda = lda.fit_transform(X_train,y_train)
X_test_lda = lda.transform(X_test)

#LDA dönüşümünden sonra
classifier_lda = LogisticRegression(random_state=0)
classifier_lda.fit(X_train_lda,y_train)

#LDA verisini tahmin et
y_pred_lda = classifier_lda.predict(X_test_lda)


#LDA sonrası / orijinal
print("LDA ve orijinal")
cm4 = confusion_matrix(y_pred_lda,y_pred)
print(cm4)



















