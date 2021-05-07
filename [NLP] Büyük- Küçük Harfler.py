# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:58:30 2020

@author: Osman
"""
import pandas as pd
comments = pd.read_csv("Restaurant_Reviews.csv",error_bad_lines=False)

import re # regular expression

comment = re.sub('[^a-zA-Z]',' ',comments['Review'][6])#substitute(replace)
comment = comment.lower() # tüm metni küçük harf yapar
comment = comment.split() #ayırarak liste yapar

