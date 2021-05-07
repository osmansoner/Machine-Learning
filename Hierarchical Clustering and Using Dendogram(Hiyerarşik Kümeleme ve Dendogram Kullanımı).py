# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 14:15:04 2020

@author: Osman
"""

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('musteriler.csv')
print(data)

x = data.iloc[:,3:].values

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3, init = 'k-means++')
kmeans.fit(x)

print(kmeans.cluster_centers_)

results = []

plt.figure(1)
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',random_state=123)
    kmeans.fit(x)
    results.append(kmeans.inertia_)

plt.plot(range(1,11),results) # sonuca göre elbow point 2 veya 4 seçilebilir.



plt.figure(2)
kmeans = KMeans(n_clusters = 4, init = 'k-means++',random_state=123)
y_tahmin = kmeans.fit_predict(x)
plt.scatter(x[y_tahmin == 0,0],x[y_tahmin == 0,1],s=100, c= 'red')
plt.scatter(x[y_tahmin == 1,0],x[y_tahmin == 1,1],s=100, c= 'blue')
plt.scatter(x[y_tahmin == 2,0],x[y_tahmin == 2,1],s=100, c= 'green')
plt.scatter(x[y_tahmin == 3,0],x[y_tahmin == 3,1],s=100, c= 'yellow')
plt.title("KMeans")
plt.show()


#Hierarchical Clustering

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters = 3,affinity='euclidean',linkage='ward')
plt.figure(3)
y_tahmin = ac.fit_predict(x)
print(y_tahmin)
plt.scatter(x[y_tahmin==0,0],x[y_tahmin==0,1],s=100, c= 'red')
plt.scatter(x[y_tahmin==1,0],x[y_tahmin==1,1],s=100, c= 'blue')
plt.scatter(x[y_tahmin==2,0],x[y_tahmin==2,1],s=100, c= 'green')
plt.scatter(x[y_tahmin == 3,0],x[y_tahmin == 3,1],s=100, c= 'yellow')

plt.title("HC")

import scipy.cluster.hierarchy as sch
plt.figure(4)
dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
plt.show()











