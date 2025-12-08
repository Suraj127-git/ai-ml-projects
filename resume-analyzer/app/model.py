import os
import re
import json
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_skills = None

def skills_path():
    base = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(base, "../data/skills_ontology.json"))

def load_skills():
    global _skills
    p = skills_path()
    if not os.path.exists(p):
        data = {"python": ["python"], "numpy": ["numpy"], "pandas": ["pandas"], "scikit-learn": ["scikit", "sklearn"], "fastapi": ["fastapi"], "tensorflow": ["tensorflow", "keras"], "pytorch": ["pytorch", "torch"]}
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            json.dump(data, f)
    with open(p) as f:
        _skills = json.load(f)

def extract_skills(text):
    t = text.lower()
    found = []
    for k, syns in _skills.items():
        for s in syns:
            if s in t:
                found.append(k)
                break
    return sorted(list(set(found)))

def score_resume(resume_text, job_text):
    v = TfidfVectorizer(max_features=5000)
    X = v.fit_transform([job_text, resume_text])
    sim = float(cosine_similarity(X[0:1], X[1:2])[0][0])
    req = extract_skills(job_text)
    have = extract_skills(resume_text)
    miss = [s for s in req if s not in have]
    cov = 0.0
    if req:
        cov = len([s for s in have if s in req]) / len(req)
    score = 100.0 * (0.5 * sim + 0.5 * cov)
    summary = "skills matched: " + ", ".join(have[:10])
    return score, have, miss, summary
