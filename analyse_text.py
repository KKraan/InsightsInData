# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 17:13:55 2019
@author: Kraan
"""
import pandas as pd
import re
from nltk.corpus import stopwords as sw
from nltk.tokenize import word_tokenize
from collections import Counter
import csv
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
import numpy as np

w2v_model = KeyedVectors.load_word2vec_format('C:\\Users\\Kees\\stack\\PYTHON\\CORPUS\\wiki\\wiki.nl.vec.bin', binary=True)

filename = 'Export_test_data_KZA.csv'
xtra_stopwords = 'added_stopwords.txt'
resultfile = 'results.csv'
nr_clusters=50


f = open(xtra_stopwords, "r")
addition = f.read().split('\n')
f.close()

stopwords = sw.words('dutch')
stopwords.extend(addition[:len(addition)-1])

data = pd.read_csv(filename, sep='\t')

data = data.loc[data['ref_specificatie'] == 'Autorisatie']
print(len(data))

allwords = []
texts = data.loc[:, 'korteomschrijving'].to_list()
pattern = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')
for text in texts:
    text = pattern.sub('', text)
    text = text.replace('-','').replace('#','').replace('.','').replace('(','').replace(')','').replace('`','').replace('?','').replace('!','').replace(':','').replace('&','').replace('>','').replace('<','').replace('@','').replace(',','').replace("'",'')
    allwords.extend(word_tokenize(text.lower()))
wordcount = Counter(allwords)

inputlist = wordcount.most_common()
first = True
for key, value in inputlist:
    if first is True:
        x = np.array([w2v_model[key]])
        first = False
    else:
        if key in w2v_model.vocab:
            x = np.append(x, [w2v_model[key]], axis=0)
        else:
            x = np.append(x, [np.zeros(300)], axis=0)

kmeans = KMeans(n_clusters=nr_clusters)
kmeans.fit(x)

newinfo = []
for i in range(len(x)):
    newinfo.append([inputlist[i][0], inputlist[i][1], kmeans.labels_[i]])

print(wordcount.most_common(50))

with open(resultfile, "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerows(newinfo)
