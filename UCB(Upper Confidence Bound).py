# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 12:31:25 2020

@author: Osman
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

data = pd.read_csv("Ads_CTR_Optimisation.csv")

#Random Selection(Rastgele Seçim)
"""
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
"""

#UCB(Upper Confidence Bound(Üst Güven Aralığı))
N = 10000 #10.000 ziyaret(visits)
d = 10 #toplam 10 reklam var ( there are totally 10 ads)
#Ri(n)
reward = [0] * d  # ilk başta bütün reklamların ödülü 0(initialize with 0 for all ads)
#Ni(n)
clicks = [0] * d #all clicks until for now 
sum_rew = 0 # toplam ödül (sum rewards)
selected = []
for n in range(0,N):
    ad = 0 #seçilen reklam(selected ad)
    max_ucb = 0
    for i in range(0,d):
        if(clicks[i] > 0):
            average = reward[i] / clicks[i]
            delta = math.sqrt(3/2* math.log(n,10)/clicks[i])
            ucb = average + delta
        else :
            ucb = N*10
            
        if max_ucb < ucb: #max'tan büyük ucb çıktı
            max_ucb = ucb
            ad = i
    selected.append(ad)
    clicks[ad] = clicks[ad] +1 
    reward_new = data.values[n,ad] # verilerde n. satır 1 ise öd8ül 1
    reward[ad] = reward[ad] + reward_new
    sum_rew = sum_rew + reward_new
print("Toplam ödül")
print(sum_rew)

plt.hist(selected)
plt.show()





























