import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "customers.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")

def ensure_data():
    if os.path.exists(DATA_PATH):
        return
    sys.path.append(os.path.join(BASE_DIR, "data"))
    from generate_data import generate_data
    df = generate_data(300)
    df.to_csv(DATA_PATH, index=False)

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def prepare_X(df):
    df = pd.get_dummies(df, columns=["Gender"], drop_first=True)
    if "Gender_Male" not in df.columns:
        df["Gender_Male"] = 0
    X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)", "Gender_Male"]].values
    return X

def train():
    ensure_data()
    df = load_data()
    X = prepare_X(df)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(Xs)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    labels = pd.Series(kmeans.labels_, name="cluster")
    out = pd.concat([df.reset_index(drop=True), labels], axis=1)
    out.to_csv(os.path.join(MODEL_DIR, "cluster_assignments.csv"), index=False)

if __name__ == "__main__":
    train()
