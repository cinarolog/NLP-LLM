from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

classifiers = {
    "MNB": MultinomialNB(),
    "BNB": BernoulliNB(),
    "Logistic Regression": LogisticRegression()
}

vectorizers = {
    "cv1": CountVectorizer(stop_words='english'),
    "cv2": CountVectorizer(ngram_range=(1, 2), binary=True, stop_words='english'),
    "tfidf1": TfidfVectorizer(stop_words='english'),
    "tfidf2": TfidfVectorizer(ngram_range=(1, 2), binary=True, stop_words='english')
}

def train_all_models(x_train, x_test, y_train, y_test):

    results = []

    for vec_name, vectorizer in vectorizers.items():

        x_train_vec = vectorizer.fit_transform(x_train)
        x_test_vec = vectorizer.transform(x_test)

        for clf_name, model in classifiers.items():

            model.fit(x_train_vec, y_train)
            y_test_pred = model.predict(x_test_vec)

            accuracy = accuracy_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred, pos_label="Positive", zero_division=1)
            recall = recall_score(y_test, y_test_pred, pos_label="Positive", zero_division=1)
            f1 = f1_score(y_test, y_test_pred, pos_label="Positive", zero_division=1)
            cm = confusion_matrix(y_test, y_test_pred)

            results.append({
                            "Vectorizer": vec_name,
                            "Classifier": clf_name,
                            "Accuracy": accuracy,
                            "Precision": precision,
                            "Recall": recall,
                            "F1 Score": f1,
                            "Confusion Matrix": cm })
            
            print(f"{vec_name} - {clf_name}:, Accuracy = {accuracy:.2f}, Precision = {precision:.2f}, Recall = {recall:.2f}, F1 Score = {f1:.2f}")
    return pd.DataFrame(results)
