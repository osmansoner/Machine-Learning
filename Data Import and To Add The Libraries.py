# -*- coding: utf-8 -*-
"""
Created on Mon May 11 04:27:41 2020

@author: Osman
"""

#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#codes
#to load the data

datas = pd.read_csv("veriler.csv")

print(datas)

height = datas[['boy']]

print(height)

heightAndWeight = datas[['boy','kilo']]

print(heightAndWeight)

x = 10
print(x)

class insan:
    boy = 1.78
    def kosma_hizi(self,hiz):
        return hiz + 7
    
osman = insan()
print(osman.boy)
print(osman.kosma_hizi(27))

l = [2,4,6,11,13] # liste gösterimi(an example for a list)