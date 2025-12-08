"""
Energy Consumption Forecasting System API
FastAPI-based API for energy consumption prediction using LSTM, Prophet, and XGBoost
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import uvicorn
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from .model import EnergyConsumptionModel
from .schemas import (
    EnergyDataPoint, ForecastRequest, ForecastResponse,
    BatchForecastRequest, BatchForecastResponse, ModelInfo,
    ModelComparisonRequest, ModelComparisonResponse,
    EnergyEfficiencyRequest, EnergyEfficiencyResponse,
    AnomalyDetectionRequest, AnomalyDetectionResponse,
    HealthResponse, BuildingType, ModelType
)

# Initialize FastAPI app
app = FastAPI(
    title="Energy Consumption Forecasting System",
    description="API for energy consumption prediction using LSTM, Prophet, and XGBoost",
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

# Global model instances
models = {
    'lstm': None,
    'prophet': None,
    'xgboost': None,
    'random_forest': None,
    'gradient_boosting': None
}

def load_models():
    """Load pre-trained models"""
    model_files = {
        'lstm': 'energy_lstm_model.h5',
        'prophet': 'energy_prophet_model.pkl',
        'xgboost': 'energy_xgboost_model.pkl',
        'random_forest': 'energy_rf_model.pkl',
        'gradient_boosting': 'energy_gb_model.pkl'
    }
    
    for model_type, filename in model_files.items():
        try:
            if os.path.exists(filename):
                if model_type == 'lstm':
                    # LSTM models are loaded differently
                    models[model_type] = True  # Placeholder
                    print(f"LSTM model file found: {filename}")
                else:
                    models[model_type] = joblib.load(filename)
                    print(f"Loaded {model_type} model")
            else:
                print(f"Model file {filename} not found for {model_type}")
        except Exception as e:
            print(f"Error loading {model_type} model: {e}")

# Load models on startup
load_models()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Energy Consumption Forecasting System API",
        "version": "1.0.0",
        "status": "active",
        "available_models": list(models.keys())
    }

@app.post("/forecast", response_model=ForecastResponse)
async def forecast_energy_consumption(request: ForecastRequest):
    """
    Forecast energy consumption for the specified number of hours
    
    Args:
        request: Forecast request with historical data and parameters
        
    Returns:
        Energy consumption forecast with confidence intervals
    """
    try:
        model_type = request.model_type.value.lower()
        
        if model_type not in models:
            raise HTTPException(status_code=400, detail=f"Model {model_type} not available")
        
        # Convert historical data to DataFrame
        historical_df = pd.DataFrame([
            {
                'timestamp': item.timestamp,
                'energy_consumption_kwh': item.energy_consumption_kwh,
                'temperature_celsius': item.temperature_celsius,
                'humidity_percent': item.humidity_percent,
                'occupancy_rate': item.occupancy_rate,
                'weather_condition': item.weather_condition,
                'is_holiday': item.is_holiday
            }
            for item in request.historical_data
        ])
        
        historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'])
        
        # Initialize model
        energy_model = EnergyConsumptionModel()
        
        # Train model with provided data
        if model_type == 'xgboost':
            training_results = energy_model.train_xgboost_model(historical_df)
        elif model_type == 'prophet':
            training_results = energy_model.train_prophet_model(historical_df)
        elif model_type == 'lstm':
            training_results = energy_model.train_lstm_model(historical_df)
        else:
            raise HTTPException(status_code=400, detail=f"Model {model_type} not implemented")
        
        # Generate forecast
        if model_type == 'xgboost':
            forecast_values, metadata = energy_model.forecast_xgboost(historical_df, request.forecast_hours)
        elif model_type == 'prophet':
            forecast_values, metadata = energy_model.forecast_prophet(request.forecast_hours)
        elif model_type == 'lstm':
            forecast_values, metadata = energy_model.forecast_lstm(historical_df, request.forecast_hours)
        
        # Create forecast timestamps
        last_timestamp = historical_df['timestamp'].iloc[-1]
        forecast_timestamps = [last_timestamp + timedelta(hours=i+1) for i in range(request.forecast_hours)]
        
        # Create forecast data points
        forecast_data = []
        for i, (timestamp, consumption) in enumerate(zip(forecast_timestamps, forecast_values)):
            forecast_data.append(EnergyDataPoint(
                timestamp=timestamp.isoformat(),
                energy_consumption_kwh=float(consumption),
                temperature_celsius=None,  # Future weather unknown
                humidity_percent=None,
                occupancy_rate=None,
                weather_condition=None,
                is_holiday=None
            ))
        
        # Calculate summary statistics
        total_forecast_consumption = float(np.sum(forecast_values))
        average_hourly_consumption = float(np.mean(forecast_values))
        peak_consumption_hour = int(np.argmax(forecast_values))
        
        # Prepare confidence intervals if available
        confidence_intervals = None
        if 'confidence_intervals' in metadata and request.include_confidence_intervals:
            confidence_intervals = {
                'lower': metadata['confidence_intervals']['lower'].tolist(),
                'upper': metadata['confidence_intervals']['upper'].tolist()
            }
        
        # Prepare model performance metrics
        model_performance = {
            'mae': training_results.get('mae', 0.0),
            'rmse': training_results.get('rmse', 0.0),
            'mape': training_results.get('mape', 0.0)
        }
        
        return ForecastResponse(
            forecast=forecast_data,
            confidence_intervals=confidence_intervals,
            model_performance=model_performance,
            total_forecast_consumption=total_forecast_consumption,
            average_hourly_consumption=average_hourly_consumption,
            peak_consumption_hour=peak_consumption_hour,
            model_type=model_type,
            forecast_timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}")

@app.post("/forecast/{model_type}", response_model=ForecastResponse)
async def forecast_with_model(model_type: str, request: ForecastRequest):
    """
    Forecast energy consumption using a specific model
    
    Args:
        model_type: Type of model to use (lstm, prophet, xgboost)
        request: Forecast request with historical data
        
    Returns:
        Energy consumption forecast
    """
    try:
        if model_type.lower() not in models:
            raise HTTPException(status_code=400, detail=f"Model {model_type} not available")
        
        # Update request with specific model type
        request.model_type = ModelType(model_type.upper())
        return await forecast_energy_consumption(request)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}")

@app.post("/batch-forecast", response_model=BatchForecastResponse)
async def batch_forecast_energy_consumption(request: BatchForecastRequest):
    """
    Forecast energy consumption for multiple buildings
    
    Args:
        request: Batch forecast request with multiple datasets
        
    Returns:
        Batch energy consumption forecasts
    """
    try:
        start_time = datetime.now()
        forecasts = []
        
        for i, (historical_data, building_type) in enumerate(zip(request.historical_data_list, request.building_types)):
            # Create individual forecast request
            individual_request = ForecastRequest(
                historical_data=historical_data,
                forecast_hours=request.forecast_hours,
                model_type=request.model_type,
                building_type=building_type
            )
            
            # Get forecast for this building
            forecast_response = await forecast_energy_consumption(individual_request)
            forecasts.append(forecast_response)
        
        # Calculate summary statistics
        total_consumptions = [f.total_forecast_consumption for f in forecasts]
        avg_consumptions = [f.average_hourly_consumption for f in forecasts]
        
        summary = {
            "total_buildings": len(forecasts),
            "total_forecasted_consumption": sum(total_consumptions),
            "average_consumption_per_building": np.mean(total_consumptions),
            "min_consumption": min(total_consumptions),
            "max_consumption": max(total_consumptions),
            "avg_hourly_consumption": np.mean(avg_consumptions)
        }
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return BatchForecastResponse(
            forecasts=forecasts,
            summary=summary,
            processing_time_seconds=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch forecast error: {str(e)}")

@app.post("/compare-models", response_model=ModelComparisonResponse)
async def compare_forecasting_models(request: ModelComparisonRequest):
    """
    Compare different forecasting models on the same dataset
    
    Args:
        request: Model comparison request
        
    Returns:
        Model comparison results
    """
    try:
        comparison_results = []
        
        # Test each model type
        model_types = [ModelType.XGBOOST, ModelType.PROPHET, ModelType.LSTM]
        
        for model_type in model_types:
            try:
                # Create forecast request for this model
                forecast_request = ForecastRequest(
                    historical_data=request.historical_data,
                    forecast_hours=request.test_hours,
                    model_type=model_type,
                    building_type=request.building_type
                )
                
                # Get forecast
                forecast_response = await forecast_energy_consumption(forecast_request)
                
                # Create model info
                model_info = ModelInfo(
                    model_type=model_type.value.lower(),
                    training_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    mae=forecast_response.model_performance.get('mae', 0.0),
                    rmse=forecast_response.model_performance.get('rmse', 0.0),
                    mape=forecast_response.model_performance.get('mape', 0.0),
                    training_samples=len(request.historical_data),
                    feature_importance=None
                )
                
                comparison_results.append(model_info)
                
            except Exception as e:
                print(f"Error testing {model_type.value}: {e}")
                continue
        
        # Find best model (lowest MAPE)
        if comparison_results:
            best_model = min(comparison_results, key=lambda x: x.mape)
            recommendation = f"Best model: {best_model.model_type} with MAPE: {best_model.mape:.2%}"
        else:
            best_model = None
            recommendation = "No models could be successfully tested"
        
        return ModelComparisonResponse(
            comparison_results=comparison_results,
            best_model=best_model.model_type if best_model else "none",
            recommendation=recommendation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model comparison error: {str(e)}")

@app.post("/efficiency-analysis", response_model=EnergyEfficiencyResponse)
async def analyze_energy_efficiency(request: EnergyEfficiencyRequest):
    """
    Analyze energy efficiency and provide recommendations
    
    Args:
        request: Energy efficiency analysis request
        
    Returns:
        Energy efficiency analysis results
    """
    try:
        # Convert historical data to DataFrame
        df = pd.DataFrame([
            {
                'timestamp': item.timestamp,
                'energy_consumption_kwh': item.energy_consumption_kwh,
                'temperature_celsius': item.temperature_celsius,
                'occupancy_rate': item.occupancy_rate
            }
            for item in request.historical_data
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Calculate baseline consumption (last N days)
        baseline_days = request.baseline_period_days
        baseline_data = df.tail(baseline_days * 24)  # Assume hourly data
        baseline_consumption = baseline_data['energy_consumption_kwh'].sum()
        
        # Calculate current consumption (recent period)
        current_data = df.tail(7 * 24)  # Last week
        current_consumption = current_data['energy_consumption_kwh'].sum()
        
        # Calculate efficiency score (0-100)
        efficiency_score = 100.0
        if current_consumption > baseline_consumption:
            efficiency_score = max(0, 100 - ((current_consumption - baseline_consumption) / baseline_consumption * 100))
        
        # Calculate savings potential
        savings_potential_kwh = max(0, current_consumption - baseline_consumption)
        savings_potential_percent = (savings_potential_kwh / current_consumption * 100) if current_consumption > 0 else 0
        
        # Identify peak and off-peak hours
        hourly_avg = current_data.groupby(current_data['timestamp'].dt.hour)['energy_consumption_kwh'].mean()
        peak_hours = hourly_avg.nlargest(3).index.tolist()
        off_peak_hours = hourly_avg.nsmallest(3).index.tolist()
        
        # Generate recommendations
        recommendations = []
        
        if efficiency_score < 70:
            recommendations.append("Consider implementing energy management system")
            recommendations.append("Schedule equipment maintenance to improve efficiency")
        
        if savings_potential_percent > 10:
            recommendations.append(f"Potential savings of {savings_potential_percent:.1f}% identified")
            recommendations.append("Review equipment operating schedules")
        
        recommendations.append(f"Shift non-critical loads to off-peak hours: {off_peak_hours}")
        recommendations.append("Monitor peak consumption hours for optimization opportunities")
        
        return EnergyEfficiencyResponse(
            efficiency_score=efficiency_score,
            baseline_consumption=baseline_consumption,
            current_consumption=current_consumption,
            savings_potential_kwh=savings_potential_kwh,
            savings_potential_percent=savings_potential_percent,
            recommendations=recommendations,
            peak_hours=peak_hours,
            off_peak_hours=off_peak_hours
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Efficiency analysis error: {str(e)}')

@app.post("/anomaly-detection", response_model=AnomalyDetectionResponse)
async def detect_energy_anomalies(request: AnomalyDetectionRequest):
    """
    Detect anomalies in energy consumption data
    
    Args:
        request: Anomaly detection request
        
    Returns:
        Anomaly detection results
    """
    try:
        # Convert historical data to DataFrame
        df = pd.DataFrame([
            {
                'timestamp': item.timestamp,
                'energy_consumption_kwh': item.energy_consumption_kwh,
                'temperature_celsius': item.temperature_celsius,
                'occupancy_rate': item.occupancy_rate
            }
            for item in request.historical_data
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Simple anomaly detection using statistical methods
        consumption_values = df['energy_consumption_kwh'].values
        
        # Calculate statistics
        mean_consumption = np.mean(consumption_values)
        std_consumption = np.std(consumption_values)
        
        # Define anomaly thresholds
        lower_threshold = mean_consumption - (3 * std_consumption * request.sensitivity)
        upper_threshold = mean_consumption + (3 * std_consumption * request.sensitivity)
        
        # Detect anomalies
        anomalies = []
        for _, row in df.iterrows():
            consumption = row['energy_consumption_kwh']
            if consumption < lower_threshold or consumption > upper_threshold:
                # Find original data point
                original_point = next(item for item in request.historical_data 
                                   if item.timestamp == row['timestamp'].isoformat())
                anomalies.append(original_point)
        
        # Calculate statistics
        anomaly_count = len(anomalies)
        anomaly_percentage = (anomaly_count / len(df)) * 100
        
        # Generate recommendations
        recommendations = []
        
        if anomaly_percentage > 5:
            recommendations.append("High anomaly rate detected - investigate equipment issues")
            recommendations.append("Consider implementing real-time monitoring system")
        
        if anomaly_count > 0:
            recommendations.append(f"{anomaly_count} anomalies detected at {anomaly_percentage:.1f}% rate")
            recommendations.append("Review data quality and sensor calibration")
        else:
            recommendations.append("No anomalies detected - normal operation")
        
        recommendations.append("Continue monitoring for unusual consumption patterns")
        
        return AnomalyDetectionResponse(
            anomalies=anomalies,
            anomaly_count=anomaly_count,
            anomaly_percentage=anomaly_percentage,
            normal_consumption_range={
                "lower": lower_threshold,
                "upper": upper_threshold,
                "mean": mean_consumption
            },
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anomaly detection error: {str(e)}")

@app.get("/models")
async def get_available_models():
    """Get list of available forecasting models"""
    return {
        "available_models": list(models.keys()),
        "loaded_models": [model for model, instance in models.items() if instance is not None]
    }

@app.get("/model-info/{model_type}", response_model=ModelInfo)
async def get_model_info(model_type: str):
    """Get information about a specific model"""
    if model_type.lower() not in models:
        raise HTTPException(status_code=404, detail=f'Model {model_type} not found')
    
    if not models[model_type.lower()]:
        raise HTTPException(status_code=503, detail=f'Model {model_type} not loaded')
    
    # Return basic model info (could be enhanced with actual model metrics)
    return ModelInfo(
        model_type=model_type,
        training_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        mae=0.0,  # Placeholder - would come from actual evaluation
        rmse=0.0,   # Placeholder - would come from actual evaluation
        mape=0.0,  # Placeholder - would come from actual evaluation
        training_samples=0,  # Placeholder - would come from actual training
        feature_importance=None  # Placeholder - would come from actual evaluation
    )

@app.post("/generate-sample-data")
async def generate_sample_energy_data(building_type: str = "office", days: int = 30):
    """
    Generate sample energy consumption data for testing
    
    Args:
        building_type: Type of building (office, residential, industrial, commercial)
        days: Number of days to generate
        
    Returns:
        Sample energy consumption data
    """
    try:
        model = EnergyConsumptionModel()
        sample_df = model.generate_synthetic_energy_data(n_days=days, building_type=building_type)
        
        # Convert to response format
        sample_data = []
        for _, row in sample_df.iterrows():
            sample_data.append(EnergyDataPoint(
                timestamp=row['timestamp'].isoformat(),
                energy_consumption_kwh=row['energy_consumption_kwh'],
                temperature_celsius=row['temperature_celsius'],
                humidity_percent=row['humidity_percent'],
                occupancy_rate=row['occupancy_rate'],
                weather_condition=row['weather_condition'],
                is_holiday=bool(row['is_holiday'])
            ))
        
        # Calculate summary statistics
        total_consumption = sample_df['energy_consumption_kwh'].sum()
        avg_consumption = sample_df['energy_consumption_kwh'].mean()
        peak_consumption = sample_df['energy_consumption_kwh'].max()
        
        # Identify peak hours
        hourly_avg = sample_df.groupby('hour_of_day')['energy_consumption_kwh'].mean()
        peak_hours = hourly_avg.nlargest(3).index.tolist()
        
        return {
            "sample_data": sample_data,
            "count": len(sample_data),
            "summary": {
                "total_consumption_kwh": total_consumption,
                "average_consumption_kwh": avg_consumption,
                "peak_consumption_kwh": peak_consumption,
                "building_type": building_type,
                "days_generated": days,
                "peak_hours": peak_hours
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data generation error: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        available_models=list(models.keys()),
        loaded_models=[model for model, instance in models.items() if instance is not None],
        timestamp=datetime.now().isoformat()
    )

# Training endpoint (for development/testing)
@app.post("/train/{model_type}")
async def train_energy_model(model_type: str, building_type: str = "office"):
    """
    Train an energy consumption forecasting model (for development)
    
    Args:
        model_type: Type of model to train (lstm, prophet, xgboost)
        building_type: Type of building for training data
    """
    try:
        print(f"Training {model_type} model for {building_type} building...")
        
        # Generate training data
        model = EnergyConsumptionModel()
        df = model.generate_synthetic_energy_data(n_days=365, building_type=building_type)
        
        # Train specific model
        if model_type == 'xgboost':
            results = model.train_xgboost_model(df)
            model.save_model('energy_xgboost_model.pkl', 'xgboost')
        elif model_type == 'prophet':
            results = model.train_prophet_model(df)
            model.save_model('energy_prophet_model.pkl', 'prophet')
        elif model_type == 'lstm':
            results = model.train_lstm_model(df)
            model.save_model('energy_lstm_model', 'lstm')
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")
        
        # Reload models
        load_models()
        
        return {
            "message": f"{model_type} model trained successfully",
            "model_type": model_type,
            "building_type": building_type,
            "training_samples": len(df),
            "performance_metrics": {
                "mae": results.get('mae', 0.0),
                "rmse": results.get('rmse', 0.0),
                "mape": results.get('mape', 0.0)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)