from fastapi import FastAPI
from app.schemas import SentimentRequest, SentimentResponse
from app.model import load_model, predict

app = FastAPI(title="Sentiment Analysis Service")

@app.on_event("startup")
async def startup():
    load_model()

@app.get("/")
def root():
    return {"message": "Sentiment Analysis Service"}

@app.post("/predict", response_model=SentimentResponse)
def pred(req: SentimentRequest):
    label, prob = predict(req.text)
    return SentimentResponse(label=label, probability=prob)
