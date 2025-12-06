from pydantic import BaseModel
from typing import List

class DigitRequest(BaseModel):
    pixels: List[List[float]]

class DigitResponse(BaseModel):
    digit: int
    probs: List[float]
