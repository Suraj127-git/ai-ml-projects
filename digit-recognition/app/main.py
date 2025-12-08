from fastapi import FastAPI
from app.schemas import DigitRequest, DigitResponse
from app.model import load_model, predict

app = FastAPI(title="Digit Recognition API")

@app.on_event("startup")
async def startup():
    load_model()

@app.get("/")
def root():
    return {"message": "Digit Recognition API"}

@app.post("/predict", response_model=DigitResponse)
def pred(req: DigitRequest):
    d, probs = predict(req.pixels)
    return DigitResponse(digit=d, probs=probs)
