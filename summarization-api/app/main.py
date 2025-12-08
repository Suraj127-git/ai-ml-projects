from fastapi import FastAPI
from app.schemas import SummarizeRequest, SummarizeResponse
from app.model import load_model, summarize

app = FastAPI(title="Text Summarization API")

@app.on_event("startup")
async def startup():
    load_model()

@app.get("/")
def root():
    return {"message": "Text Summarization API"}

@app.post("/summarize", response_model=SummarizeResponse)
def sum_api(req: SummarizeRequest):
    s = summarize(req.text)
    return SummarizeResponse(summary=s)
