#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 10:26:26 2022

@author: prathamy
"""

import re
import numpy as np
import pandas as pd
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.adapt import MLkNN
from collections import Counter
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import skmultilearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import hamming_loss, accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report




data_size = 1000
df = pd.read_csv("/home/prathamy/Documents/Documents/ACADS/SEM1/FML/processed_train_data_new.csv",nrows=data_size)
dft = df['Tags']
dft_list = list(dft)
dfText = df['Text']
# dfText_list = list[dfText]

for i in range(data_size) :
    temp = re.sub("[\[\],]","",dft_list[i])
    dft[i] = temp

for i in range(data_size) :
    temp = re.sub("[\[\],']","",df['Text'][i])
    dfText[i] = temp
    
# print(type(df['Text'][0]))
# print(df['Text'][0][0])

vectorizer = CountVectorizer(tokenizer = lambda x: x.split(), binary='true', min_df=1)
vectorizer.fit(dft) #tags
# print("Vocabulary: ", vectorizer.vocabulary_)
# print(len(vectorizer.vocabulary_))
vector = vectorizer.transform(dft)
# print(vector.toarray())
vector = vector.toarray()
vector = vector.astype(float)

tags_dict = vectorizer.vocabulary_

tags=[0 for i in range(0,100)]

for i in tags_dict :
    tags[tags_dict[i]] = i

sampledf = pd.DataFrame(vector, columns = tags)
# print(sampledf)

"""
2. make the contiuous string into dataframe, maybe try doing it inplace.
3. conactenate with sampledf 
4. see if it the final df is in desired format
"""


# final = pd.concat([dfText, sampledf], axis=1)
# print(final)

tfidf = TfidfVectorizer()
y = np.array(sampledf)
Xfeatures = tfidf.fit_transform(dfText).toarray()
X_train, X_test, y_train, y_test = train_test_split(Xfeatures,y,test_size=0.3,random_state=42)


mlknn = MLkNN(k=10)
mlknn.fit(X_train, y_train)
pred = mlknn.predict(X_test)
pred = pred.toarray()

print(hamming_loss(y_test,pred))


print(classification_report(y_test, pred))