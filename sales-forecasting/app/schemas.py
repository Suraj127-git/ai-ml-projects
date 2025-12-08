from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import date, datetime

class SalesData(BaseModel):
    """Input schema for sales data point"""
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    sales: float = Field(..., description="Sales amount")
    
class ForecastRequest(BaseModel):
    """Request schema for sales forecasting"""
    historical_data: List[SalesData] = Field(..., description="Historical sales data")
    forecast_days: int = Field(default=30, ge=1, le=365, description="Number of days to forecast")
    model_type: str = Field(default="prophet", description="Model type: prophet, arima, or linear_regression")
    
class ForecastResponse(BaseModel):
    """Response schema for sales forecast"""
    forecast: List[SalesData] = Field(..., description="Predicted sales for future dates")
    confidence_intervals: Optional[Dict[str, List[float]]] = Field(None, description="Confidence intervals")
    model_performance: Optional[Dict[str, float]] = Field(None, description="Model performance metrics")
    
class ModelInfo(BaseModel):
    """Schema for model information"""
    model_type: str
    training_date: str
    rmse: float
    mae: float
    mape: float
    training_samples: int