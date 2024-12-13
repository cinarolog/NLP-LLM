#%%
import nltk
import numpy as np
import pandas as pd
import re
import string
from nltk import SnowballStemmer
from textblob import TextBlob
from nltk.tokenize import MWETokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer

from ML_Pipeline.utils import load_data 
from ML_Pipeline.preprocess import positive_negative_review,preprocess_text,stemming_review
from ML_Pipeline.model import train_all_models

# import data
file_path="C:/Users/cinar/Desktop/NLP/input/coffee.csv"
data=load_data(file_path)

#positive-negative reviews
data=positive_negative_review(data)
data=preprocess_text(data)
print(data.head())

#stemming
sbs = SnowballStemmer(language='english')
data=stemming_review(data)

polarity_scores = []
for review in data.reviews:
    text = TextBlob(review)
    sentiment = text.sentiment.polarity
    polarity_scores.append(sentiment)

data['polarity'] = polarity_scores

def determine_sentiment(polarity_score):
    if polarity_score > 0:
        return 'Positive'
    elif polarity_score < 0:
        return 'Negative'
    else:
        return 'Neutral'

data['new_sentiment'] = data.polarity.apply(determine_sentiment)

#train-test-split
X = data.reviews
y = data.sentiment

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train.shape, y_train.shape)

#model
results=train_all_models(X_train, X_test, y_train, y_test)

results_df = results.set_index(["Vectorizer", "Classifier"])
results_df







# %%
