import os
import joblib

_pipe = None

def _path():
    base = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(base, "../model/sentiment.pkl"))

def load_model():
    global _pipe
    _pipe = joblib.load(_path())

def predict(text: str):
    proba = _pipe.predict_proba([text])[0]
    idx = int(proba.argmax())
    label = _pipe.classes_[idx]
    return label, float(proba[idx])
