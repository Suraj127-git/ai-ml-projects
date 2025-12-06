from fastapi import FastAPI
from app.schemas import UserRequest, RecommendationResponse
from app.model import load_model, recommend

app = FastAPI(title="Movie Recommender API")

@app.on_event("startup")
async def startup():
    load_model()

@app.get("/")
def root():
    return {"message": "Movie Recommender API"}

@app.post("/recommend", response_model=RecommendationResponse)
def rec(req: UserRequest):
    items = recommend(req.user_id, req.k)
    return RecommendationResponse(items=items)
