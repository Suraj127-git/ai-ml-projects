from pydantic import BaseModel
from typing import Optional, List
from datetime import date
from enum import Enum

class ModelName(str, Enum):
    XGBOOST = "xgboost"
    BG_NBD = "bg_nbd"

class CustomerData(BaseModel):
    customer_id: str
    recency: int  # days since last purchase
    frequency: int  # number of purchases
    monetary_value: float  # total spent
    tenure: int  # days since first purchase
    avg_order_value: float
    days_between_purchases: float
    total_orders: int
    age: Optional[int] = None
    gender: Optional[str] = None
    country: Optional[str] = None

class CLVPrediction(BaseModel):
    customer_id: str
    predicted_clv: float
    model_used: str
    confidence_score: Optional[float] = None
    prediction_date: date
    expected_purchases_next_30_days: Optional[float] = None
    expected_purchases_next_90_days: Optional[float] = None
    expected_purchases_next_365_days: Optional[float] = None

class ModelInfo(BaseModel):
    model_name: str
    model_type: str
    features: List[str]
    training_date: Optional[date] = None
    model_performance: Optional[dict] = None
    version: str = "1.0.0"

class TrainingResponse(BaseModel):
    message: str
    model_type: str
    training_samples: int
    model_performance: dict

class HealthResponse(BaseModel):
    status: str
    available_models: List[str]
    loaded_models: int
    timestamp: str