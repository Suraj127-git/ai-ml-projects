from fastapi import FastAPI
import subprocess
import os

app = FastAPI(title="Auto Retraining API")

@app.get("/")
def root():
    return {"message": "Auto Retraining API"}

@app.post("/retrain")
def retrain():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    p = os.path.join(base, "sentiment-service", "notebooks", "train_sentiment.py")
    subprocess.run(["python", p], check=False)
    return {"status": "triggered"}
