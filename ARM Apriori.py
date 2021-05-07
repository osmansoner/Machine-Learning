# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 12:47:10 2020

@author: Osman
"""

import pandas as pd

data = pd.read_csv('sepet.csv',header= None)


#Hocanın çözümü; çok fazla non değer var ve biraz karışık 
"""

t = []

for i in range(0,7501):
    t.append([str(data.values[i,j]) for j in range(0,20)])
    
from apyori import apriori 
rules = apriori(t,min_support=0.01, min_confidence=0.02,min_lift = 3,max_length = 2)

print(list(rules))
"""

cleaned_list = []

for x in t:
    clist = []
    for z in x:
        if str(z) != 'nan':
            clist.append(z)
    cleaned_list.append(clist)
    
from apyori import apriori,TransactionManager
rules = apriori(cleaned_list,min_support=0.01,min_confidence = 0.02,min_lift=3)
rules_list = list(rules)

for item in rules_list:
    
    base_items = [x for x in item[2][0][0]]
    add_item, = item[2][0][1]
    print("Rule: " + " + ".join(base_items) + " -> " + str(add_item))
    print("Support: " + str(item[1])) 
    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")
























    




    