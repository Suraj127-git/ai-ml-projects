import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(BASE, "data", "interactions.csv")
MODEL_DIR = os.path.join(BASE, "model")

def load_data():
    return pd.read_csv(DATA)

def build_item_similarity(df):
    users = df['user'].unique()
    items = df['product'].unique()
    ui = pd.DataFrame(0.0, index=users, columns=items)
    for _, r in df.iterrows():
        ui.loc[r['user'], r['product']] = r['events']
    mat = ui.values.T
    sim = cosine_similarity(mat)
    return ui, pd.DataFrame(sim, index=items, columns=items)

def train():
    df = load_data()
    ui, item_sim = build_item_similarity(df)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(item_sim, os.path.join(MODEL_DIR, "item_sim.pkl"))
    joblib.dump(ui, os.path.join(MODEL_DIR, "user_item.pkl"))

if __name__ == "__main__":
    train()
