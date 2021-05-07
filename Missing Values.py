# -*- coding: utf-8 -*-
"""
Created on Tue May 12 01:56:07 2020

@author: Osman
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

missing_values_in_data = pd.read_csv("eksikveriler.csv")
print(missing_values_in_data)

#eksik veriler
#missing values

#sci(science) - kit - learn
from sklearn.preprocessing import Imputer #eski versiyonda kullanılıyormuş.Kaldırıldı.
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy ="mean")

#kayıp değerler için NaN yerine mean yani ortalama değer konulacak.
#We're gonna put the mean value of the column instead of NaN value.

Age = missing_values_in_data.iloc[:,1:4].values
print(Age)
imputer = imputer.fit(Age[:,1:4])
Age[:,1:4] = imputer.transform(Age[:,1:4])
print(Age)
