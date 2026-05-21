from flask import Flask, render_template, request
app = Flask(__name__)
import pandas as pd
fake=pd.read_csv("Fake.csv")

real=pd.read_csv("True.csv")

fake.head(5)

real.head(5)

fake.describe()

fake.isnull().sum()

real.isnull().sum()

fake['title'] = fake['title'].str.lower().str.strip()
fake['text'] = fake['text'].str.lower().str.strip()
real['title'] = real['title'].str.lower().str.strip()
real['text'] = real['text'].str.lower().str.strip()

fake['label']=0
real['label']=1

df=pd.concat([fake,real],ignore_index=True).sample(frac=1, random_state=42)

df

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import numpy as np
import string
import re

for pkg in ["stopwords", "wordnet", "punkt"]:
    try:
        nltk.download(pkg, quiet=True)
    except:
        pass

"""Cleaning"""

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = text.translate(str.maketrans('', '', string.punctuation))

    text = re.sub(r"\d+", "", text)

    text = text.strip()

    return text

df["clean_text"] = df["text"].apply(clean_text)

df.head()



"""Feature Engineering"""

def build_features(df):
    df = df.copy()
    df["content"] = (df.get("title", "").fillna("") + " " +
                     df.get("text", "").fillna(""))
    df["clean_content"] = df["content"].apply(clean_text)
    return df

df.tail(3)

from sklearn.model_selection import train_test_split
X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

"""TF-IDF VECTORIZER"""

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

"""Logistic *Regression*"""

lr = LogisticRegression()

lr.fit(X_train_tfidf, y_train)

lr_pred = lr.predict(X_test_tfidf)

print("Logistic Regression")

print("Accuracy:", accuracy_score(y_test, lr_pred))
print("Precision:", precision_score(y_test, lr_pred))
print("Recall:", recall_score(y_test, lr_pred))
print("F1 Score:", f1_score(y_test, lr_pred))

print(classification_report(y_test, lr_pred))

"""Naive Bayes"""

nb = MultinomialNB()

nb.fit(X_train_tfidf, y_train)

nb_pred = nb.predict(X_test_tfidf)

print("Naive Bayes")

print("Accuracy:", accuracy_score(y_test, nb_pred))
print("Precision:", precision_score(y_test, nb_pred))
print("Recall:", recall_score(y_test, nb_pred))
print("F1 Score:", f1_score(y_test, nb_pred))

def predict_news(news):

    news = clean_text(news)

    vector = vectorizer.transform([news])

    prediction = lr.predict(vector)

    if prediction[0] == 1:
        return "Real News"
    else:
        return "Fake News"

@app.route('/', methods=['GET', 'POST'])

def home():

    result = ""

    if request.method == 'POST':

        news = request.form['news']

        result = predict_news(news)

    return render_template('index.html', result=result)


if __name__ == '__main__':

    app.run(debug=True)
