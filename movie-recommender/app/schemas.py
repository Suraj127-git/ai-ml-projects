from pydantic import BaseModel
from typing import List

class UserRequest(BaseModel):
    user_id: int
    k: int = 5

class RecommendationResponse(BaseModel):
    items: List[int]
