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

filename = 'Export_test_data_KZA.csv'
xtra_stopwords = 'added_stopwords.txt'

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
print(wordcount.most_common(50))

with open("results.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerows(wordcount.most_common(50))
