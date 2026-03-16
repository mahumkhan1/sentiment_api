# imports
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import pickle
import os

# load data
# print("downloading data")
# fetch_20newsgroups(download_if_missing=True, remove=('headers','footers','quotes'))
# print("downloaded data")

good, bad = ['rec.sport.hockey', 'rec.sport.baseball', 'sci.space'], ['soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast']

good_words = fetch_20newsgroups(subset='all', categories=good, shuffle=True, random_state=42)
bad_words = fetch_20newsgroups(subset='all', categories=bad, shuffle=True, random_state=42)

texts = good_words.data + bad_words.data
labels = [1] * len(good_words.data) + [0] * len(bad_words.data)  # 1=positive, 0=negative
print("generated data")

# split data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, stratify=labels, random_state=42)
print("split data")

# preprocess data / build pipeline
vectorizer = TfidfVectorizer(max_features=20000,
                        ngram_range=(1, 2),   # unigrams + bigrams
                        min_df=2,
                        stop_words='english')
classifier = LogisticRegression(C=1.0,
                        max_iter=1000,
                        solver='lbfgs')

pipeline = Pipeline([
    ('tfidf', vectorizer),
    ('clf', classifier)
])
print("preprocessed data")

# train
pipeline.fit(X_train, y_train)  
print("trained model")

# evaluate
y_pred = pipeline.predict(X_test)
accuracy_score = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy_score}\nReport:\n{report}")

# save model
os.makedirs("sentiment_api/model", exist_ok=True)
with open('sentiment_api/model/model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
