from pydantic import BaseModel
from typing import Optional

class CustomerRequest(BaseModel):
    gender: str
    age: int
    annual_income_k: int
    spending_score: int

class CustomerResponse(BaseModel):
    cluster: int
    id: Optional[int] = None
