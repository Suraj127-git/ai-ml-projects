from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import uvicorn
import os
import pandas as pd
import numpy as np

from .model import DemandForecastingModel
from .schemas import (
    ProductData, ForecastRequest, ForecastResponse, BatchForecastRequest, 
    BatchForecastResponse, TrainingRequest, TrainingResponse, ModelInfo, 
    HealthResponse, ModelName, DemandPattern
)

# Initialize FastAPI app
app = FastAPI(
    title="Product Demand Forecasting API",
    description="API for forecasting product demand using ARIMA, Prophet, LSTM, and XGBoost models",
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
demand_model = DemandForecastingModel()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Product Demand Forecasting API",
        "version": "1.0.0",
        "available_models": ["arima", "prophet", "lstm", "xgboost"],
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        available_models=["arima", "prophet", "lstm", "xgboost"],
        loaded_models=len(demand_model.arima_models) + len(demand_model.prophet_models) + 
                      len(demand_model.lstm_models) + len(demand_model.xgboost_models),
        total_forecasts=len(demand_model.model_performance),
        timestamp=datetime.now()
    )

@app.post("/forecast", response_model=ForecastResponse)
async def forecast_demand(request: ForecastRequest):
    """Generate demand forecast for a product"""
    try:
        # Convert request data to DataFrame
        df = pd.DataFrame([{
            'product_id': request.product_id,
            'date': item.date,
            'demand': 0,  # Will be filled from historical data
            'price': item.price,
            'promotion': item.promotion,
            'seasonality': item.seasonality,
            'holiday': item.holiday,
            'stock_level': item.stock_level
        } for item in request.historical_data])
        
        # Generate forecast
        forecast_result = demand_model.forecast_demand(
            product_id=request.product_id,
            model_type=request.model_type.value,
            periods=request.forecast_periods,
            historical_data=df
        )
        
        return ForecastResponse(
            product_id=request.product_id,
            model_type=request.model_type.value,
            forecast_periods=request.forecast_periods,
            forecast=forecast_result['forecast'],
            confidence_intervals=forecast_result.get('confidence_intervals'),
            model_performance=demand_model.model_performance.get(f"{request.product_id}_{request.model_type.value}"),
            forecast_date=datetime.now().date()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecasting error: {str(e)}")

@app.post("/forecast/batch", response_model=BatchForecastResponse)
async def batch_forecast(request: BatchForecastRequest):
    """Generate batch forecasts for multiple products"""
    start_time = datetime.now()
    forecasts = []
    
    try:
        for product_request in request.products:
            # Convert request data to DataFrame
            df = pd.DataFrame([{
                'product_id': product_request.product_id,
                'date': item.date,
                'demand': 0,
                'price': item.price,
                'promotion': item.promotion,
                'seasonality': item.seasonality,
                'holiday': item.holiday,
                'stock_level': item.stock_level
            } for item in product_request.historical_data])
            
            # Generate forecast
            forecast_result = demand_model.forecast_demand(
                product_id=product_request.product_id,
                model_type=product_request.model_type.value,
                periods=product_request.forecast_periods,
                historical_data=df
            )
            
            forecast_response = ForecastResponse(
                product_id=product_request.product_id,
                model_type=product_request.model_type.value,
                forecast_periods=product_request.forecast_periods,
                forecast=forecast_result['forecast'],
                confidence_intervals=forecast_result.get('confidence_intervals'),
                model_performance=demand_model.model_performance.get(f"{product_request.product_id}_{product_request.model_type.value}"),
                forecast_date=datetime.now().date()
            )
            
            forecasts.append(forecast_response)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return BatchForecastResponse(
            forecasts=forecasts,
            total_products=len(forecasts),
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch forecasting error: {str(e)}")

@app.post("/train/{model_type}")
async def train_model(model_type: ModelName, request: TrainingRequest):
    """Train a demand forecasting model"""
    try:
        # Convert training data to DataFrame
        df = pd.DataFrame([{
            'product_id': item.product_id,
            'date': item.date,
            'demand': item.demand,
            'price': item.price,
            'promotion': item.promotion,
            'seasonality': item.seasonality,
            'holiday': item.holiday,
            'stock_level': item.stock_level
        } for item in request.training_data])
        
        # Get unique products
        products = df['product_id'].unique()
        
        training_results = {}
        total_samples = len(df)
        
        for product_id in products:
            product_data = df[df['product_id'] == product_id]
            
            if model_type == ModelName.ARIMA:
                result = demand_model.train_arima_model(df, product_id)
            elif model_type == ModelName.PROPHET:
                result = demand_model.train_prophet_model(df, product_id)
            elif model_type == ModelName.LSTM:
                result = demand_model.train_lstm_model(df, product_id)
            elif model_type == ModelName.XGBOOST:
                result = demand_model.train_xgboost_model(df, product_id)
            else:
                raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")
            
            training_results[product_id] = result
        
        # Calculate average performance
        avg_performance = {}
        if training_results:
            all_metrics = list(training_results.values())[0].keys()
            for metric in all_metrics:
                if metric not in ['training_samples', 'test_samples', 'error']:
                    values = [result.get(metric, 0) for result in training_results.values() if metric in result]
                    if values:
                        avg_performance[metric] = np.mean(values)
        
        return TrainingResponse(
            message=f"{model_type.value} model trained successfully for {len(products)} products",
            model_type=model_type.value,
            training_samples=total_samples,
            validation_samples=int(total_samples * request.validation_split),
            model_performance=avg_performance,
            training_duration=0.0  # Will be calculated in real implementation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@app.get("/model/info/{product_id}", response_model=ModelInfo)
async def get_model_info(product_id: str):
    """Get information about trained models for a specific product"""
    try:
        info = demand_model.get_model_info(product_id)
        
        # Find best performing model
        best_model = None
        best_mape = float('inf')
        
        for model_type, performance in info['model_performance'].items():
            if 'mape' in performance and performance['mape'] < best_mape:
                best_mape = performance['mape']
                best_model = model_type
        
        return ModelInfo(
            model_name=f"demand_forecast_{product_id}",
            model_type=best_model or "not_trained",
            version="1.0.0",
            features=['price', 'promotion', 'holiday', 'day_of_week', 'month', 'lag_features'],
            model_performance=info['model_performance'].get(best_model) if best_model else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@app.get("/model/performance")
async def get_model_performance():
    """Get detailed model performance metrics"""
    try:
        return {
            "model_performance": demand_model.model_performance,
            "models_available": {
                "arima": len(demand_model.arima_models),
                "prophet": len(demand_model.prophet_models),
                "lstm": len(demand_model.lstm_models),
                "xgboost": len(demand_model.xgboost_models)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model performance: {str(e)}")

@app.post("/generate-sample-data")
async def generate_sample_data(n_products: int = 5, n_days: int = 365):
    """Generate sample product demand data for testing"""
    try:
        df = demand_model.generate_synthetic_demand_data(n_products, n_days)
        
        # Convert to ProductData objects
        products = []
        for _, row in df.iterrows():
            product = ProductData(
                product_id=row['product_id'],
                date=row['date'].date(),
                demand=float(row['demand']),
                price=float(row['price']),
                promotion=int(row['promotion']),
                seasonality=float(row['seasonality']),
                holiday=int(row['holiday']),
                stock_level=int(row['stock_level'])
            )
            products.append(product)
        
        # Group by product
        products_by_id = {}
        for product in products:
            if product.product_id not in products_by_id:
                products_by_id[product.product_id] = []
            products_by_id[product.product_id].append(product)
        
        return {
            "sample_products": products_by_id,
            "total_products": len(products_by_id),
            "total_records": len(products),
            "summary_stats": {
                "avg_demand": float(df['demand'].mean()),
                "avg_price": float(df['price'].mean()),
                "promotion_rate": float(df['promotion'].mean()),
                "holiday_rate": float(df['holiday'].mean())
            },
            "date_range": {
                "start": df['date'].min().isoformat(),
                "end": df['date'].max().isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating sample data: {str(e)}")

@app.post("/analyze-demand-pattern")
async def analyze_demand_pattern(historical_data: List[ProductData]):
    """Analyze demand patterns in historical data"""
    try:
        df = pd.DataFrame([{
            'date': item.date,
            'demand': item.demand
        } for item in historical_data])
        
        df = df.sort_values('date')
        df.set_index('date', inplace=True)
        
        # Calculate trend
        demand_values = df['demand'].values
        x = np.arange(len(demand_values))
        trend_slope = np.polyfit(x, demand_values, 1)[0]
        
        if trend_slope > 0.01:
            trend = "increasing"
        elif trend_slope < -0.01:
            trend = "decreasing"
        else:
            trend = "stable"
        
        # Calculate seasonality
        if len(demand_values) >= 365:
            # Simple seasonality detection
            monthly_avg = df.groupby(df.index.month)['demand'].mean()
            seasonal_variation = monthly_avg.std() / monthly_avg.mean()
            
            if seasonal_variation > 0.2:
                seasonality = "strong"
            elif seasonal_variation > 0.1:
                seasonality = "moderate"
            else:
                seasonality = "weak"
        else:
            seasonality = "insufficient_data"
        
        # Calculate volatility
        volatility = np.std(demand_values) / np.mean(demand_values)
        
        if volatility > 0.5:
            volatility_level = "high"
        elif volatility > 0.2:
            volatility_level = "medium"
        else:
            volatility_level = "low"
        
        return DemandPattern(
            trend=trend,
            seasonality=seasonality,
            volatility=volatility_level,
            growth_rate=float(trend_slope),
            seasonal_strength=float(seasonal_variation) if seasonality != "insufficient_data" else 0.0
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing demand pattern: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)