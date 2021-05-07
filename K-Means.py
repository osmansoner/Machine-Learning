# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 12:20:38 2020

@author: Osman
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('musteriler.csv')
print(data)

x = data.iloc[:,3:].values

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3, init = 'k-means++')
kmeans.fit(x)

print(kmeans.cluster_centers_)

results = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',random_state=123)
    kmeans.fit(x)
    results.append(kmeans.inertia_)

plt.plot(range(1,11),results) # sonuca göre elbow point 2,3 veya 4 seçilebilir. 










