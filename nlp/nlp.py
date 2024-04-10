# -*- coding: utf-8 -*-
"""NLP.ipynb

Automatically generated by Colab.

"""

!pip install nltk

import numpy as np
import pandas as pd
import nltk
import math
from string import punctuation
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import chain
import re

# nltk.download()
nltk.download('stopwords')

data = pd.read_csv('disaster-tweets.csv')
print(data)

data.dropna(inplace=True, axis=1)
data.info()

# Cleaning
def preprocess_text(data):
    stemmer = PorterStemmer()
    stop_punc = set(stopwords.words('english')).union(set(punctuation))

    def process_text(text):
        words = wordpunct_tokenize(text)
        words_lower = map(lambda w: w.lower(), words)
        words_filtered = filter(lambda w: w not in stop_punc, words_lower)
        words_filtered = filter(lambda w: not re.match(r'^[:/]+|[.?]{2,}$', w), words_filtered)
        words_stemmed = map(stemmer.stem, words_filtered)
        return list(words_stemmed)

    return list(map(process_text, data["text"]))

cleaned = preprocess_text(data)
print(len(cleaned))

# Filtering
def flatten(data):
    return [word for sublist in data for word in sublist]

def create_vocab(cleaned):
    flattened_data = flatten(cleaned)
    word_counts = Counter(flattened_data)
    vocabulary = set(word_counts.keys())
    most_frequent = dict(sorted(word_counts.items(), key=lambda item: item[1], reverse=True))

    return list(vocabulary), most_frequent

def get_words_frequency(cleaned, most_frequent):
    for i in cleaned:
        for word in i:
            most_frequent[word] += 1

    sorted_dict = dict(sorted(most_frequent.items(), key=lambda item: -item[1]))

    cleaned_words = list(sorted_dict.keys())
    cleaned_words = cleaned_words[4:10004]

    return cleaned_words

vocab, most_frequent = create_vocab(cleaned)
cleaned_words = get_words_frequency(cleaned, most_frequent)

print(cleaned_words)
len(cleaned_words)

# BoW
def numocc_score(word, doc):
  return doc.count(word)

def create_bow_features(cleaned, cleaned_words):
    print('Creating BOW features...')
    X = np.zeros((len(cleaned), len(cleaned_words)), dtype=np.float32)
    for item_idx in range(len(cleaned)):
        item = cleaned[item_idx]
        for word_idx in range(len(cleaned_words)):
            word = cleaned_words[word_idx]
            cnt = numocc_score(word, item)
            X[item_idx][word_idx] = cnt
    print(X)
    return X

"""## Naive Bayes
* $P(C) = \frac{|Training~set~ elements~Class|}{|Training~set|}$

---

* $P(C|BoW) \sim P(C) \cdot \prod{P(Word^i|C)^{BoW[Word^i]}}$

---

* $P(Word^i|C) = \frac{occurences(Word^i, C) + \alpha}{total\_words\_number(C) + |Vocab| \cdot \alpha}$
"""

class MultinomialNaiveBayes:
  def __init__(self, classes_num, words_num, pseudocount):
    self.classes_num = classes_num
    self.words_num = words_num
    self.pseudocount = pseudocount

  def fit(self, X, Y):
    examples_num = X.shape[0]

    # P(Class) - Priors
    self.priors = np.bincount(Y) / examples_num
    print('Priors:')
    print(self.priors)

    occurences = np.zeros((self.classes_num, self.words_num))
    for i in range(examples_num):
      c = Y[i]
      for w in range(self.words_num):
        cnt = X[i][w]
        occurences[c][w] += cnt
    print('Occurences:')
    print(occurences)

    # P(X|Class) - Likelihoods
    self.like = np.zeros((self.classes_num, self.words_num))
    for c in range(self.classes_num):
      for w in range(self.words_num):
        up = occurences[c][w] + self.pseudocount
        down = np.sum(occurences[c]) + self.words_num*self.pseudocount
        self.like[c][w] = up / down
    print('Likelihoods:')
    print(self.like)

  def predict(self, bow):
    # P(Class|BoW) - Posterior
    probs = np.zeros(self.classes_num)
    for c in range(self.classes_num):
      prob = np.log(self.priors[c])
      for w in range(self.words_num):
        cnt = bow[w]
        prob += cnt * np.log(self.like[c][w])
      probs[c] = prob
    prediction = np.argmax(probs)
    return prediction

X = create_bow_features(cleaned, cleaned_words)
y = data['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  stratify=y)

model = MultinomialNaiveBayes(classes_num=2, words_num=len(cleaned_words), pseudocount=1)
model.fit(X_train, y_train)

for i in range(3):
  _, X_test, _, y_test = train_test_split(X, y, test_size=0.2,  stratify=y)
  predictions = []
  for sample in X_test:
    pred = model.predict(sample)
    predictions.append(pred)
  accuracy = accuracy_score(y_test, predictions)
  print('Accuracy: ', accuracy)

positive_comm = []
negative_comm = []


for i, target in enumerate(data['target']):
  if target == 1:
    positive_comm.append(cleaned[i])
  else:
    negative_comm.append(cleaned[i])


positive_comm_flat = list(chain.from_iterable(positive_comm))
negative_comm_flat = list(chain.from_iterable(negative_comm))


positive_word_counts = Counter(" ".join(positive_comm_flat).split())
negative_word_counts = Counter(" ".join(negative_comm_flat).split())


top_positive_words = positive_word_counts.most_common(10)
print("Top 5 words in positive tweets:", top_positive_words)


top_negative_words = negative_word_counts.most_common(5)
print("Top 5 words in negative tweets:", top_negative_words)


LR_scores = {}
for word, count in positive_word_counts.items():
    if count >= 10 and negative_word_counts[word] >= 10:
        LR_scores[word] = count / negative_word_counts[word]


top_LR_words = sorted(LR_scores.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top 5 words with highest LR scores:", top_LR_words)


bottom_LR_words = sorted(LR_scores.items(), key=lambda x: x[1])[:5]
print("Top 5 words with lowest LR scores:", bottom_LR_words)