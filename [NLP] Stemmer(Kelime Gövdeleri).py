# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 14:38:40 2020

@author: Osman
"""

import pandas as pd
comments = pd.read_csv("Restaurant_Reviews.csv",error_bad_lines=False)

import re #regular expression

import nltk #natural language toolkit
nltk.download('stopwords') 

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
stop_words.remove('not') #Duygusal kutupsallıkta karmaşıklık olmaması için not stopword den çıkarılır.
derlem = []
derlem2 = [] #'not' stopword den çıkarılırsa
for i in range(1000):
    comment = re.sub('[^a-zA-Z]',' ',comments['Review'][i])
    comment = comment.lower() 
    comment = comment.split() 
    comment = [ps.stem(kelime) for kelime in comment if not kelime in set(stopwords.words('english'))] # türkçesi de var
    comment = ' '.join(comment) #birleştirir. list i str yapar. (split in tersi gibi)
    derlem.append(comment)


#'not' stopwordden çıkarılırsa (şu an için çalışmıyor.)
for n in range(1000):
    comment2 = re.sub('[^a-zA-Z]',' ',comments['Review'][n])
    comment2 = comment.lower() 
    comment2 = comment.split() 
    comment2 = [ps.stem(kelime) for kelime in comment if not kelime in stop_words] # türkçesi de var
    comment2 = ' '.join(comment) #birleştirir. list i str yapar. (split in tersi gibi)
    derlem2.append(comment)














