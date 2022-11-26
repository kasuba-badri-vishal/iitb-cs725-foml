#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on Thu Nov  3 10:26:26 2022

@author: prathamy
"""
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import hamming_loss
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


""" To run this code, first run "pip uninstall scikit-learn -y" on terminal.
Then run "pip install scikit-learn==0.24.1" """



data_size = 25000
df = pd.read_csv("processed_train_data_new.csv", nrows=data_size)
""" Download this file with name "processed_train_data_new.csv" from the link
given : "https://drive.google.com/file/d/1V7VNvTBWgVvJYXg5czlBIp_ZGRqLMiBn/view?usp=sharing" """
""" Then run this file in the same directory as the one with this data file """
#data_size = df.shape[0]

dft = df['Tags']
dft_list = list(dft)
dfText = df['Text']
# dfText_list = list[dfText]

for i in range(data_size):
    temp = re.sub("[\[\],]", "", dft_list[i])
    dft[i] = temp

for i in range(data_size):
    temp = re.sub("[\[\],']", "", df['Text'][i])
    dfText[i] = temp

# print(type(df['Text'][0]))
# print(df['Text'][0][0])

vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), binary='true', min_df=1)
vectorizer.fit(dft)  # tags
# print("Vocabulary: ", vectorizer.vocabulary_)
# print(len(vectorizer.vocabulary_))
vector = vectorizer.transform(dft)
# print(vector.toarray())
vector = vector.toarray()
# vector = vector.astype(float)

tags_dict = vectorizer.vocabulary_

tags = [0 for i in range(0, 100)]

for i in tags_dict:
    tags[tags_dict[i]] = i

sampledf = pd.DataFrame(vector, columns=tags)
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

from scipy import sparse
x = sparse.csr_matrix(Xfeatures)
y = sparse.csr_matrix(y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# binary_rel_clf = BinaryRelevance(MultinomialNB())

# binary_rel_clf.fit(X_train, y_train)

# pred = binary_rel_clf.predict(X_test)

# pred = pred.toarray()

mlknn = MLkNN(k=31)
mlknn.fit(X_train, y_train)
pred = mlknn.predict(X_test)
pred = pred.toarray()

print(f"score : {mlknn.score(X_test,y_test)}")
print(f"hamming score : {hamming_loss(y_test, pred)}")

precision = precision_score(y_test, pred, average='micro')
recall = recall_score(y_test, pred, average='micro')

print("precision : ", precision)
print("recall : ", recall)

print(f"Classification Report : {classification_report(y_test, pred)}")
