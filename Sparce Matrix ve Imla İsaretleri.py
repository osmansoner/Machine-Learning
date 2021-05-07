# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:33:31 2020

@author: Osman
"""

import pandas as pd
comments = pd.read_csv("Restaurant_Reviews.csv",error_bad_lines=False)

import re # regular expression

comment = re.sub('[^a-zA-Z]',' ',comments['Review'][0])#substitute(replace)
comment2 = re.sub('[^a-zA-Z]',' ',comments['Review'][6])#substitute(replace)
