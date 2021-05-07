# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 11:41:51 2020

@author: Osman
"""

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Ads_CTR_Optimisation.csv")

import random 
N = 10000
d = 10
sum = 0
selected = []
for n in range(0,N):
    ad =random.randrange(d)
    selected.append(ad)
    reward = data.values[n,ad] #verilerdeki n.satır = 1 ise ödül 1
    sum = sum + reward
    
plt.hist(selected)
plt.show()