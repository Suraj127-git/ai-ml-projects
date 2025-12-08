from fastapi import FastAPI
from app.schemas import ResumeTextRequest, AnalyzeResponse
from app.model import load_skills, score_resume

app = FastAPI(title="AI Resume Analyzer")

@app.on_event("startup")
async def startup():
    load_skills()

@app.get("/")
def root():
    return {"message": "AI Resume Analyzer"}

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: ResumeTextRequest):
    score, have, miss, summary = score_resume(req.resume_text, req.job_text)
    return AnalyzeResponse(score=score, matched_skills=have, missing_skills=miss, summary=summary)
