from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class ModelType(str, Enum):
    """Available neural network models for demand forecasting"""
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    CNN_LSTM = "cnn_lstm"
    BILSTM = "bilstm"

class ForecastingStrategy(str, Enum):
    """Forecasting strategies"""
    MULTIVARIATE = "multivariate"
    UNIVARIATE = "univariate"
    SEASONAL = "seasonal"
    TREND_BASED = "trend_based"

class DemandData(BaseModel):
    """Input data for demand forecasting"""
    product_id: str = Field(..., description="Product identifier")
    timestamp: datetime = Field(..., description="Timestamp of the data point")
    demand: float = Field(..., ge=0, description="Actual demand quantity")
    price: Optional[float] = Field(None, ge=0, description="Product price")
    promotion: Optional[int] = Field(None, ge=0, le=1, description="Promotion indicator (0/1)")
    seasonality: Optional[float] = Field(None, description="Seasonality factor")
    external_factors: Optional[Dict[str, float]] = Field(None, description="External factors")

class ForecastingRequest(BaseModel):
    """Request for demand forecasting"""
    product_id: str = Field(..., description="Product identifier")
    historical_data: List[DemandData] = Field(..., description="Historical demand data")
    forecast_horizon: int = Field(..., ge=1, le=365, description="Number of periods to forecast")
    model_type: ModelType = Field(default=ModelType.LSTM, description="Neural network model type")
    strategy: ForecastingStrategy = Field(default=ForecastingStrategy.MULTIVARIATE, description="Forecasting strategy")
    sequence_length: int = Field(default=30, ge=7, le=365, description="Sequence length for training")
    confidence_level: float = Field(default=0.95, ge=0.8, le=0.99, description="Confidence level for predictions")

class ForecastingResponse(BaseModel):
    """Response from demand forecasting"""
    product_id: str = Field(..., description="Product identifier")
    forecast: List[float] = Field(..., description="Predicted demand values")
    confidence_intervals: Dict[str, List[float]] = Field(..., description="Lower and upper confidence intervals")
    model_performance: Dict[str, float] = Field(..., description="Model performance metrics")
    forecast_dates: List[datetime] = Field(..., description="Forecast dates")
    strategy_used: str = Field(..., description="Strategy used for forecasting")
    model_type: str = Field(..., description="Model type used")

class BatchForecastingRequest(BaseModel):
    """Request for batch demand forecasting"""
    products: List[ForecastingRequest] = Field(..., description="List of forecasting requests")
    parallel_processing: bool = Field(default=True, description="Enable parallel processing")

class BatchForecastingResponse(BaseModel):
    """Response from batch demand forecasting"""
    forecasts: List[ForecastingResponse] = Field(..., description="List of forecasting results")
    processing_time: float = Field(..., description="Total processing time in seconds")
    successful_count: int = Field(..., description="Number of successful forecasts")
    failed_count: int = Field(..., description="Number of failed forecasts")

class ModelTrainingRequest(BaseModel):
    """Request for model training"""
    training_data: List[DemandData] = Field(..., description="Training data")
    model_type: ModelType = Field(..., description="Model type to train")
    strategy: ForecastingStrategy = Field(..., description="Forecasting strategy")
    validation_split: float = Field(default=0.2, ge=0.1, le=0.4, description="Validation split ratio")
    epochs: int = Field(default=100, ge=10, le=1000, description="Number of training epochs")
    batch_size: int = Field(default=32, ge=8, le=256, description="Batch size for training")
    learning_rate: float = Field(default=0.001, ge=0.0001, le=0.1, description="Learning rate")

class ModelTrainingResponse(BaseModel):
    """Response from model training"""
    model_id: str = Field(..., description="Trained model identifier")
    model_type: str = Field(..., description="Model type trained")
    training_metrics: Dict[str, float] = Field(..., description="Training metrics")
    validation_metrics: Dict[str, float] = Field(..., description="Validation metrics")
    training_time: float = Field(..., description="Training time in seconds")
    model_size: int = Field(..., description="Model size in bytes")

class ModelInfo(BaseModel):
    """Model information"""
    model_id: str = Field(..., description="Model identifier")
    model_type: str = Field(..., description="Model type")
    strategy: str = Field(..., description="Forecasting strategy")
    created_at: datetime = Field(..., description="Model creation time")
    performance_metrics: Dict[str, float] = Field(..., description="Model performance metrics")
    is_active: bool = Field(..., description="Whether the model is active")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Response timestamp")
    available_models: List[str] = Field(..., description="List of available models")
    gpu_available: bool = Field(..., description="Whether GPU is available")