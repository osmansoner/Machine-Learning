# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 12:14:14 2020

@author: Osman
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

data = pd.read_csv("Ads_CTR_Optimisation.csv")

#Thompson Sampling
N = 10000
d = 10

sum = 0
selected = []
ones = [0] * d # birler
zeros = [0] * d #sıfırlar
for n in range(1,N):
    ad = 0 #seçilen reklam
    max_th =0
    for i in range(0,d):
        ran_beta = random.betavariate(ones[i] + 1 , zeros[i] + 1)#beta oluşturma
        if ran_beta > max_th:
            max_th = ran_beta
            ad = i
    selected.append(ad)
    reward = data.values[n,ad]
    if reward == 1: # reklam tıklanmışsa yani 1 se
        ones[ad] = ones[ad] + 1 # 0 ile initialize yapıldığı için 1 ekleyerek 1 yapmış oluruz.
    else : #reklam 0 sa yani tıklanmamışsa
        zeros[ad] = zeros[ad] + 1
    sum = sum + reward
print("Toplam ödül:")
print(sum)

plt.hist(selected)
plt.show()













