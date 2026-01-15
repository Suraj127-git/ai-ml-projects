from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import Dict, Any, List
import uvicorn
import os
import numpy as np
import pandas as pd

from .model import DemandForecastingNeuralModel
from .schemas import (
    ForecastingRequest, ForecastingResponse, ModelTrainingRequest,
    HealthResponse, ModelInfo, ModelType, ModelPerformance
)

# Initialize FastAPI app
app = FastAPI(
    title="Demand Forecasting with Neural Networks API",
    description="API for demand forecasting using LSTM, GRU, Transformer, CNN-LSTM, and BiLSTM models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
forecasting_model = DemandForecastingNeuralModel()

# Model storage
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    """Load models on startup if they exist"""
    lstm_path = os.path.join(MODEL_DIR, "lstm_model.pth")
    
    try:
        if os.path.exists(lstm_path):
            forecasting_model.load_model(lstm_path, "lstm")
            print("Loaded LSTM model successfully")
    except Exception as e:
        print(f"Could not load LSTM model: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Demand Forecasting with Neural Networks API",
        "version": "1.0.0",
        "available_models": ["lstm", "gru", "transformer", "cnn_lstm", "bilstm"],
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_info = forecasting_model.get_model_info()
    
    return HealthResponse(
        status="healthy",
        available_models=["lstm", "gru", "transformer", "cnn_lstm", "bilstm"],
        loaded_models=len(model_info['available_models']),
        timestamp=datetime.now().isoformat()
    )

@app.post("/forecast", response_model=ForecastingResponse)
async def forecast_demand(request: ForecastingRequest):
    """
    Forecast demand using neural network models
    
    Args:
        request: Forecasting request with historical data and parameters
    """
    try:
        # Convert historical data to list of demands
        historical_demands = [data.demand for data in request.historical_data]
        
        # Check if we have enough data
        if len(historical_demands) < forecasting_model.sequence_length:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient historical data. Need at least {forecasting_model.sequence_length} data points, got {len(historical_demands)}"
            )
        
        # Check if model is trained
        model_type_str = request.model_type.value
        if model_type_str not in forecasting_model.models:
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_type_str} not trained. Please train the model first."
            )
        
        # Generate forecasts
        forecasts = forecasting_model.forecast_demand(
            model_type_str, 
            historical_demands, 
            request.forecast_horizon
        )
        
        # Create forecast dates
        last_date = request.historical_data[-1].date
        forecast_dates = []
        for i in range(request.forecast_horizon):
            next_date = last_date + pd.Timedelta(days=i+1)
            forecast_dates.append(next_date)
        
        # Create forecast objects
        forecast_data = []
        for i, (date, demand) in enumerate(zip(forecast_dates, forecasts)):
            forecast_data.append({
                "date": date,
                "predicted_demand": demand,
                "confidence_lower": demand * 0.9,  # Simple confidence interval
                "confidence_upper": demand * 1.1
            })
        
        # Get model performance info
        model_info = forecasting_model.get_model_info()
        performance = model_info['model_performance'].get(model_type_str, {})
        
        return ForecastingResponse(
            product_id=request.product_id,
            model_type=request.model_type,
            forecast_horizon=request.forecast_horizon,
            forecasts=forecast_data,
            performance_metrics=performance,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Forecasting error: {str(e)}')

@app.post("/train/{model_type}", response_model=Dict[str, Any])
async def train_model(model_type: ModelType, n_samples: int = 1000):
    """
    Train a neural network model for demand forecasting
    
    Args:
        model_type: Type of neural network model to train
        n_samples: Number of synthetic samples to generate for training
    """
    try:
        print(f"Training {model_type} model with {n_samples} samples...")
        
        # Generate synthetic training data
        df = forecasting_model.generate_synthetic_demand_data(n_samples)
        
        # Train model based on type
        if model_type == ModelType.LSTM:
            performance = forecasting_model.train_lstm_model(df)
            model_file = "lstm_model.pth"
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Training for {model_type} not implemented yet"
            )
        
        # Save model
        model_path = os.path.join(MODEL_DIR, model_file)
        forecasting_model.save_model(model_path, model_type.value)
        
        return {
            "message": f"{model_type} model trained successfully",
            "model_type": model_type,
            "training_samples": n_samples,
            "performance_metrics": performance,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Training error: {str(e)}')

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about available models"""
    try:
        model_info = forecasting_model.get_model_info()
        
        return ModelInfo(
            model_name="Demand Forecasting Neural Networks",
            model_type="Neural Network Ensemble",
            available_models=model_info['available_models'],
            sequence_length=model_info['sequence_length'],
            device=model_info['device'],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error getting model info: {str(e)}')

@app.get("/model/performance/{model_type}", response_model=ModelPerformance)
async def get_model_performance(model_type: ModelType):
    """Get performance metrics for a specific model"""
    try:
        model_info = forecasting_model.get_model_info()
        performance = model_info['model_performance'].get(model_type.value)
        
        if not performance:
            raise HTTPException(
                status_code=404,
                detail=f"Performance data for {model_type} not found. Please train the model first."
            )
        
        return ModelPerformance(
            model_type=model_type,
            rmse=performance.get('rmse', 0),
            mae=performance.get('mae', 0),
            mape=performance.get('mape', 0),
            training_samples=performance.get('training_samples', 0),
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error getting model performance: {str(e)}')

@app.post("/generate-sample-data")
async def generate_sample_data(n_days: int = 365):
    """
    Generate sample demand data for testing
    
    Args:
        n_days: Number of days of sample data to generate
    """
    try:
        df = forecasting_model.generate_synthetic_demand_data(n_days)
        
        # Convert to list of dictionaries
        sample_data = []
        for _, row in df.iterrows():
            sample_data.append({
                "date": row['date'].strftime('%Y-%m-%d'),
                "demand": float(row['demand']),
                "product_id": row['product_id']
            })
        
        return {
            "sample_data": sample_data[:100],  # Return first 100 points
            "total_generated": len(df),
            "summary_stats": {
                "avg_demand": float(df['demand'].mean()),
                "std_demand": float(df['demand'].std()),
                "min_demand": float(df['demand'].min()),
                "max_demand": float(df['demand'].max())
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error generating sample data: {str(e)}')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)