# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 14:18:10 2020

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

#Preprocessing(Ön İşleme)
derlem = []
derlem2 = [] #'not' stopword den çıkarılırsa
for i in range(716):
    comment = re.sub('[^a-zA-Z]',' ',comments['Review'][i])
    comment = comment.lower() 
    comment = comment.split() 
    comment = [ps.stem(kelime) for kelime in comment if not kelime in set(stopwords.words('english'))] # türkçesi de var
    comment = ' '.join(comment) #birleştirir. list i str yapar. (split in tersi gibi)
    derlem.append(comment)


#'not' stopwordden çıkarılırsa (şu an için çalışmıyor.)
for n in range(716):
    comment2 = re.sub('[^a-zA-Z]',' ',comments['Review'][n])
    comment2 = comment.lower() 
    comment2 = comment.split() 
    comment2 = [ps.stem(kelime) for kelime in comment if not kelime in stop_words] # türkçesi de var
    comment2 = ' '.join(comment) #birleştirir. list i str yapar. (split in tersi gibi)
    derlem2.append(comment)



#Feature Extraction(Öznitelik Çıkarımı)
#Bag of Words(BOW)    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(derlem).toarray() #bağımsız değişken
y = comments.iloc[:,1].values  #bağımlı değişken

#Machine Learning(Makine Öğrenmesi)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

from sklearn.metrics import confussion_matrix
cm = confussion_matrix(y_test,y_pred)
print(cm)




































