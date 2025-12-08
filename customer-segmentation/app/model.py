import os
import sqlite3
import joblib
import numpy as np
import pandas as pd

_kmeans = None
_scaler = None

def _model_paths():
    base = os.path.dirname(__file__)
    kmeans_path = os.path.join(base, "../model/kmeans.pkl")
    scaler_path = os.path.join(base, "../model/scaler.pkl")
    return os.path.abspath(kmeans_path), os.path.abspath(scaler_path)

def load_model():
    global _kmeans, _scaler
    kmeans_path, scaler_path = _model_paths()
    _kmeans = joblib.load(kmeans_path)
    _scaler = joblib.load(scaler_path)
    return _kmeans, _scaler

def ensure_db():
    db_path = os.path.join(os.path.dirname(__file__), "../data/segments.db")
    conn = sqlite3.connect(os.path.abspath(db_path))
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP,
            gender TEXT,
            age INTEGER,
            annual_income_k INTEGER,
            spending_score INTEGER,
            cluster INTEGER
        )
        """
    )
    conn.commit()
    conn.close()

def _to_df(payload):
    d = {
        "Gender": [payload.gender],
        "Age": [payload.age],
        "Annual Income (k$)": [payload.annual_income_k],
        "Spending Score (1-100)": [payload.spending_score],
    }
    return pd.DataFrame(d)

def _preprocess(df):
    df = pd.get_dummies(df, columns=["Gender"], drop_first=True)
    for col in ["Gender_Male"]:
        if col not in df.columns:
            df[col] = 0
    X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)", "Gender_Male"]].values
    Xs = _scaler.transform(X)
    return Xs

def predict_cluster(payload):
    df = _to_df(payload)
    Xs = _preprocess(df)
    c = int(_kmeans.predict(Xs)[0])
    return c

def save_result(payload, cluster):
    db_path = os.path.join(os.path.dirname(__file__), "../data/segments.db")
    conn = sqlite3.connect(os.path.abspath(db_path))
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO segments (gender, age, annual_income_k, spending_score, cluster) VALUES (?, ?, ?, ?, ?)",
        (payload.gender, payload.age, payload.annual_income_k, payload.spending_score, cluster),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id

def get_results(limit=50):
    db_path = os.path.join(os.path.dirname(__file__), "../data/segments.db")
    conn = sqlite3.connect(os.path.abspath(db_path))
    cur = conn.cursor()
    cur.execute(
        "SELECT id, ts, gender, age, annual_income_k, spending_score, cluster FROM segments ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows

def get_cluster_summary():
    db_path = os.path.join(os.path.dirname(__file__), "../data/segments.db")
    conn = sqlite3.connect(os.path.abspath(db_path))
    cur = conn.cursor()
    cur.execute("SELECT cluster, COUNT(*) FROM segments GROUP BY cluster ORDER BY cluster")
    rows = cur.fetchall()
    conn.close()
    return rows
