from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

class StockData(BaseModel):
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL, GOOGL, MSFT)")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    open_price: float = Field(..., ge=0, description="Opening price")
    high_price: float = Field(..., ge=0, description="High price")
    low_price: float = Field(..., ge=0, description="Low price")
    close_price: float = Field(..., ge=0, description="Closing price")
    volume: int = Field(..., ge=0, description="Trading volume")
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "date": "2024-01-15",
                "open_price": 150.25,
                "high_price": 152.80,
                "low_price": 149.50,
                "close_price": 151.75,
                "volume": 85000000
            }
        }

class StockPredictionRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol to predict")
    days_ahead: int = Field(default=1, ge=1, le=30, description="Number of days ahead to predict")
    model_type: str = Field(default="lstm", description="Model type: lstm, linear_regression, or random_forest")
    include_features: bool = Field(default=True, description="Include technical indicators in prediction")
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "days_ahead": 1,
                "model_type": "lstm",
                "include_features": True
            }
        }

class StockPrediction(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    predicted_direction: str = Field(..., description="Predicted price movement direction (Up/Down)")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    probability_up: float = Field(..., ge=0, le=1, description="Probability of price going up")
    probability_down: float = Field(..., ge=0, le=1, description="Probability of price going down")
    predicted_price: Optional[float] = Field(None, description="Predicted price (if available)")
    current_price: float = Field(..., description="Current price")
    model_used: str = Field(..., description="Model used for prediction")
    prediction_date: str = Field(..., description="Date of prediction")
    target_date: str = Field(..., description="Target date for prediction")
    key_features: Optional[Dict[str, float]] = Field(None, description="Key features used in prediction")
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "predicted_direction": "Up",
                "confidence": 0.75,
                "probability_up": 0.75,
                "probability_down": 0.25,
                "predicted_price": 155.20,
                "current_price": 151.75,
                "model_used": "lstm",
                "prediction_date": "2024-01-15",
                "target_date": "2024-01-16",
                "key_features": {
                    "rsi": 45.2,
                    "macd": 1.25,
                    "volume_ratio": 1.15
                }
            }
        }

class BatchStockPrediction(BaseModel):
    predictions: List[StockPrediction]
    summary: Dict[str, int] = Field(..., description="Summary of predictions")
    average_confidence: float = Field(..., description="Average confidence across all predictions")

class ModelInfo(BaseModel):
    model_type: str = Field(..., description="Type of model")
    training_date: str = Field(..., description="Date when model was trained")
    accuracy: Optional[float] = Field(None, description="Model accuracy on test set")
    features_used: List[str] = Field(..., description="Features used by the model")
    model_parameters: Dict[str, any] = Field(..., description="Model parameters")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "lstm",
                "training_date": "2024-01-01",
                "accuracy": 0.68,
                "features_used": ["close_price", "volume", "rsi", "macd"],
                "model_parameters": {
                    "hidden_units": 64,
                    "dropout": 0.2,
                    "sequence_length": 60
                }
            }
        }

class TrainingRequest(BaseModel):
    symbols: List[str] = Field(..., description="List of stock symbols to train on")
    model_type: str = Field(default="lstm", description="Model type to train")
    sequence_length: int = Field(default=60, ge=10, le=200, description="Sequence length for time series")
    test_split: float = Field(default=0.2, ge=0.1, le=0.4, description="Test split ratio")
    epochs: int = Field(default=50, ge=10, le=200, description="Number of training epochs")
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbols": ["AAPL", "GOOGL", "MSFT"],
                "model_type": "lstm",
                "sequence_length": 60,
                "test_split": 0.2,
                "epochs": 50
            }
        }

class TrainingResponse(BaseModel):
    message: str = Field(..., description="Training status message")
    models_trained: List[str] = Field(..., description="List of models trained")
    training_time: float = Field(..., description="Training time in seconds")
    best_accuracy: float = Field(..., description="Best model accuracy achieved")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Models trained successfully",
                "models_trained": ["AAPL_lstm", "GOOGL_lstm", "MSFT_lstm"],
                "training_time": 125.5,
                "best_accuracy": 0.72
            }
        }