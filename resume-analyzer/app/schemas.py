from pydantic import BaseModel
from typing import Optional, List

class ResumeTextRequest(BaseModel):
    resume_text: str
    job_text: str

class AnalyzeResponse(BaseModel):
    score: float
    matched_skills: List[str]
    missing_skills: List[str]
    summary: str
