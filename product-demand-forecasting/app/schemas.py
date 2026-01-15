from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import date, datetime
from enum import Enum

class ModelName(str, Enum):
    ARIMA = "arima"
    PROPHET = "prophet"
    LSTM = "lstm"
    XGBOOST = "xgboost"

class ForecastPeriod(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

class ProductData(BaseModel):
    product_id: str
    date: date
    demand: float
    price: Optional[float] = None
    promotion: Optional[int] = Field(None, ge=0, le=1)
    seasonality: Optional[float] = None
    holiday: Optional[int] = Field(None, ge=0, le=1)
    stock_level: Optional[int] = None

class HistoricalData(BaseModel):
    product_id: str
    data: List[ProductData]
    
    class Config:
        json_encoders = {
            date: lambda v: v.isoformat()
        }

class ForecastRequest(BaseModel):
    product_id: str
    historical_data: List[ProductData]
    forecast_periods: int = Field(default=30, ge=1, le=365)
    model_type: ModelName = ModelName.PROPHET
    include_confidence_interval: bool = True
    confidence_level: float = Field(default=0.95, ge=0.8, le=0.99)

class ForecastResponse(BaseModel):
    product_id: str
    model_type: str
    forecast_periods: int
    forecast: List[Dict[str, Any]]
    confidence_intervals: Optional[Dict[str, List[float]]] = None
    model_performance: Optional[Dict[str, float]] = None
    forecast_date: date
    last_training_date: Optional[date] = None
    
    class Config:
        json_encoders = {
            date: lambda v: v.isoformat()
        }

class BatchForecastRequest(BaseModel):
    products: List[ForecastRequest]
    
class BatchForecastResponse(BaseModel):
    forecasts: List[ForecastResponse]
    total_products: int
    processing_time: float

class TrainingRequest(BaseModel):
    training_data: List[ProductData]
    model_type: ModelName
    validation_split: float = Field(default=0.2, ge=0.1, le=0.4)
    hyperparameter_tuning: bool = False

class TrainingResponse(BaseModel):
    message: str
    model_type: str
    training_samples: int
    validation_samples: int
    model_performance: Dict[str, float]
    training_duration: float
    model_parameters: Optional[Dict[str, Any]] = None

class ModelInfo(BaseModel):
    model_name: str
    model_type: str
    version: str
    features: List[str]
    training_date: Optional[date] = None
    last_updated: Optional[date] = None
    model_performance: Optional[Dict[str, float]] = None
    hyperparameters: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    available_models: List[str]
    loaded_models: int
    total_forecasts: int
    timestamp: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class DemandPattern(BaseModel):
    trend: str
    seasonality: str
    volatility: str
    growth_rate: float
    seasonal_strength: float