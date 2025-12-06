import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(BASE, "data", "reviews.csv")
MODEL_DIR = os.path.join(BASE, "model")

def train():
    df = pd.read_csv(DATA)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("clf", LogisticRegression(max_iter=100)),
    ])
    pipe.fit(df.text, df.label)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipe, os.path.join(MODEL_DIR, "sentiment.pkl"))

if __name__ == "__main__":
    train()
