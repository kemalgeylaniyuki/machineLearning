# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 18:56:12 2022

@author: kemal
"""

import pandas as pd

#%% Read twitter data

data = pd.read_csv(r"gender_classifier.csv", encoding = "latin1")
data = pd.concat([data.gender,data.description,],axis=1) 
data.dropna(axis=0,inplace=True) # NAN değerler sil

data.gender = [1 if i == "female" else 0 for i in data.gender]

#%% Data cleaning
# Regular Expression : RE

import re

first_description = data.description[4] 
# [^a-zA-Z] : a dan z ye ve A dan Z ye olmayan harfleri boşluk ile değiştir.
description = re.sub("[^a-zA-Z]", " ", first_description) 
description = description.lower() # Tüm harfleri küçük yap

#%% Stopwords : (irrelavent words) : gereksiz kelimeler

import nltk # natural language tool kit
nltk.download("punkt") # corpus diye bir kalsore indiriliyor
from nltk.corpus import stopwords  # sonra ben corpus klasorunden import ediyorum

# description = description.split()
# split yerine tokenizer kullanabiliriz.
description = nltk.word_tokenize(description)
# split kullanırsak "shouldn't " gibi kelimeler "should" ve "not" diye ikiye ayrılmaz ama word_tokenize() kullanirsak ayrilir.

# %%
# greksiz kelimeleri cikar

description = [word for word in description if not word in set(stopwords.words("english"))]

#%%  Lemmatization EX : Loved -> Love yapmak, çekimlerden kurtulmak.

import nltk as nlp

lemma = nlp.WordNetLemmatizer() 

description = [lemma.lemmatize(word) for word in description]

description = " ".join(description)

#%% Data Cleaning

description_list = []

for description in data.description:
    description = re.sub("[^a-zA-Z]", " ", description) 
    # [^a-zA-Z] : a dan z ye ve A dan Z ye olmayan harfleri boşluk ile değiştir.
    description = description.lower() # Tüm harfleri küçük yap
    description = nltk.word_tokenize(description)
    # description = description.split()
    # split yerine tokenizer kullanabiliriz.
    # split kullanırsak "shouldn't " gibi kelimeler "should" ve "not" diye ikiye ayrılmaz.
    description = [lemma.lemmatize(word) for word in description] # Kelimeyi çekimlerden kurtar
    description = " ".join(description) # Kelimeleri araya boşluk koyarak birleştir.
    description_list.append(description)
    
# %% # %% bag of words

from sklearn.feature_extraction.text import CountVectorizer

max_features = 500

count_vectorizer = CountVectorizer(max_features = max_features, stop_words = "english")

sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()

print("En sık kullanılan {} kelimeler : {} ".format(max_features,count_vectorizer.get_feature_names()))

#%% Train test split

y = data.iloc[:,0].values # male or female classes
x = sparce_matrix

# train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)

#%% Naive Bayes

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train,y_train)

#%% prediction

y_pred = nb.predict(x_test)

print("Accuracy : ", nb.score(x_test, y_test))
    



























