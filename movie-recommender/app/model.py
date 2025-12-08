import os
import joblib
import numpy as np

_item_sim = None
_user_item = None

def _paths():
    base = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(base, "../model/item_sim.pkl")), os.path.abspath(os.path.join(base, "../model/user_item.pkl"))

def load_model():
    global _item_sim, _user_item
    p1, p2 = _paths()
    _item_sim = joblib.load(p1)
    _user_item = joblib.load(p2)

def recommend(user_id: int, k: int = 5):
    if user_id not in _user_item.index:
        return []
    user_ratings = _user_item.loc[user_id]
    scored = {}
    for item in _user_item.columns:
        if user_ratings[item] > 0:
            sims = _item_sim.loc[item]
            for j, s in sims.items():
                if user_ratings[j] == 0:
                    scored[j] = scored.get(j, 0.0) + s * user_ratings[item]
    ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
    return [i for i, _ in ranked[:k]]
