from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime, timedelta
import os
import pandas as pd

from app.schemas import ForecastRequest, ForecastResponse, ModelInfo, SalesData
from app.model import SalesForecastingModel

# Initialize FastAPI app
app = FastAPI(
    title="Sales Forecasting API",
    description="API for forecasting sales using time series models (ARIMA, Prophet, Linear Regression)",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
models = {}
model_files = {
    'prophet': 'sales_forecast_prophet.pkl',
    'arima': 'sales_forecast_arima.pkl',
    'linear_regression': 'sales_forecast_linear.pkl'
}

# Load models on startup
@app.on_event("startup")
async def startup_event():
    """Load the sales forecasting models on startup"""
    global models
    
    for model_type, model_file in model_files.items():
        try:
            if os.path.exists(model_file):
                model = SalesForecastingModel()
                model.load_model(model_file)
                models[model_type] = model
                print(f"{model_type} model loaded successfully!")
            else:
                print(f"{model_file} not found. Please train the models first.")
        except Exception as e:
            print(f"Error loading {model_type} model: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Sales Forecasting API",
        "version": "1.0.0",
        "status": "active",
        "available_models": list(models.keys())
    }

@app.post("/forecast", response_model=ForecastResponse)
async def forecast_sales(request: ForecastRequest):
    """
    Forecast sales for the specified number of days
    
    Args:
        request: Forecast request with historical data and parameters
        
    Returns:
        Sales forecast with confidence intervals
    """
    try:
        model_type = request.model_type.lower()
        
        if model_type not in models:
            raise HTTPException(status_code=400, detail=f"Model {model_type} not available")
        
        if not models[model_type]:
            raise HTTPException(status_code=503, detail=f'Model {model_type} not loaded')
        
        # Convert historical data to DataFrame
        historical_df = pd.DataFrame([
            {'date': item.date, 'sales': item.sales}
            for item in request.historical_data
        ])
        
        # Generate forecast based on model type
        if model_type == 'prophet':
            forecast_data, confidence_intervals = models[model_type].forecast_prophet(request.forecast_days)
        elif model_type == 'arima':
            forecast_data, confidence_intervals = models[model_type].forecast_arima(request.forecast_days)
        elif model_type == 'linear_regression':
            # Prepare data for linear regression
            prepared_df = models[model_type].prepare_data(historical_df)
            forecast_data, confidence_intervals = models[model_type].forecast_linear_regression(prepared_df, request.forecast_days)
        
        # Convert forecast data to response format
        forecast_items = [
            SalesData(date=item['date'], sales=item['sales'])
            for item in forecast_data
        ]
        
        return ForecastResponse(
            forecast=forecast_items,
            confidence_intervals=confidence_intervals,
            model_performance=None  # Could add model evaluation metrics here
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}")

@app.post("/forecast/{model_type}")
async def forecast_with_model(model_type: str, request: ForecastRequest):
    """
    Forecast sales using a specific model
    
    Args:
        model_type: Type of model to use (prophet, arima, linear_regression)
        request: Forecast request with historical data
        
    Returns:
        Sales forecast
    """
    try:
        if model_type.lower() not in models:
            raise HTTPException(status_code=400, detail=f"Model {model_type} not available")
        
        # Update request with specific model type
        request.model_type = model_type
        return await forecast_sales(request)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}")


@app.get("/models")
async def get_available_models():
    """Get list of available forecasting models"""
    return {
        "available_models": list(models.keys()),
        "loaded_models": [model for model, instance in models.items() if instance is not None]
    }

@app.get("/model-info/{model_type}")
async def get_model_info(model_type: str):
    """Get information about a specific model"""
    if model_type.lower() not in models:
        raise HTTPException(status_code=404, detail=f'Model {model_type} not found')
    
    if not models[model_type.lower()]:
        raise HTTPException(status_code=503, detail=f"Model {model_type} not loaded")
    
    # Return basic model info (could be enhanced with actual model metrics)
    return ModelInfo(
        model_type=model_type,
        training_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        rmse=0.0,  # Placeholder - would come from actual evaluation
        mae=0.0,   # Placeholder - would come from actual evaluation
        mape=0.0,  # Placeholder - would come from actual evaluation
        training_samples=0  # Placeholder - would come from actual training
    )

@app.post("/generate-sample-data")
async def generate_sample_data(days: int = 365):
    """
    Generate sample sales data for testing
    
    Args:
        days: Number of days to generate
        
    Returns:
        Sample sales data
    """
    try:
        model = SalesForecastingModel()
        sample_df = model.generate_synthetic_sales_data(n_days=days)
        
        # Convert to response format
        sample_data = [
            SalesData(date=row['date'].strftime('%Y-%m-%d'), sales=row['sales'])
            for _, row in sample_df.iterrows()
        ]
        
        return {"sample_data": sample_data, "count": len(sample_data)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data generation error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "available_models": list(models.keys()),
        "loaded_models": len([m for m in models.values() if m is not None]),
        "timestamp": datetime.now().isoformat()
    }

# Training endpoint (for development/testing)
@app.post("/train/{model_type}")
async def train_model(model_type: str):
    """
    Train a sales forecasting model (for development)
    
    Args:
        model_type: Type of model to train (prophet, arima, linear_regression)
    """
    try:
        print(f"Training {model_type} model...")
        
        # Generate training data
        model = SalesForecastingModel()
        df = model.generate_synthetic_sales_data(n_days=730)
        df_prepared = model.prepare_data(df)
        
        # Train specific model
        if model_type == 'prophet':
            model.train_prophet_model(df_prepared)
            model.save_model('sales_forecast_prophet.pkl', 'prophet')
        elif model_type == 'arima':
            model.train_arima_model(df_prepared)
            model.save_model('sales_forecast_arima.pkl', 'arima')
        elif model_type == 'linear_regression':
            model.train_linear_regression_model(df_prepared)
            model.save_model('sales_forecast_linear.pkl', 'linear_regression')
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")
        
        return {
            "message": f"{model_type} model trained successfully",
            "model_type": model_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)