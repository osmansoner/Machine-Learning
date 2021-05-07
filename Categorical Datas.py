# -*- coding: utf-8 -*-
"""
Created on Thu May 14 01:35:40 2020

@author: Osman
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

losted_datas = pd.read_csv("eksikveriler.csv")
print(losted_datas)

#we have to seperate the columns that we wanna categorize
#kategorize etmek istediğimiz sütunu ayrımamız gerekiyor. 
country = losted_datas.iloc[:,0:1].values
print(country)

label_encoder = LabelEncoder()

#değiştirip yerleştirme işlemi aynı anda yaptık ama sadece sütün olarak görünüyor şu anda
#we changed and fitted the values 
country[:,0] = label_encoder.fit_transform(country[:,0])
print(country)

ohe = OneHotEncoder(categorical_features='all')
country = ohe.fit_transform(country).toarray()
print(country)
