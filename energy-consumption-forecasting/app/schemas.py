"""
Pydantic schemas for Energy Consumption Forecasting System API
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime, date
from enum import Enum

class BuildingType(str, Enum):
    """Building types for energy consumption forecasting"""
    OFFICE = "office"
    RESIDENTIAL = "residential"
    INDUSTRIAL = "industrial"
    COMMERCIAL = "commercial"

class WeatherCondition(str, Enum):
    """Weather conditions"""
    SUNNY = "sunny"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    SNOWY = "snowy"
    FOGGY = "foggy"
    WINDY = "windy"

class ModelType(str, Enum):
    """Available forecasting models"""
    LSTM = "lstm"
    PROPHET = "prophet"
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"

class EnergyDataPoint(BaseModel):
    """Input schema for energy consumption data point"""
    timestamp: str = Field(..., description="Timestamp in ISO format")
    energy_consumption_kwh: float = Field(..., ge=0, description="Energy consumption in kWh")
    temperature_celsius: Optional[float] = Field(None, description="Temperature in Celsius")
    humidity_percent: Optional[float] = Field(None, ge=0, le=100, description="Humidity percentage")
    occupancy_rate: Optional[float] = Field(None, ge=0, le=1, description="Occupancy rate (0-1)")
    weather_condition: Optional[WeatherCondition] = Field(None, description="Weather condition")
    is_holiday: Optional[bool] = Field(False, description="Whether it's a holiday")

class ForecastRequest(BaseModel):
    """Request schema for energy consumption forecasting"""
    historical_data: List[EnergyDataPoint] = Field(..., description="Historical energy consumption data")
    forecast_hours: int = Field(default=24, ge=1, le=168, description="Number of hours to forecast")
    model_type: ModelType = Field(default=ModelType.XGBOOST, description="Model type to use")
    building_type: BuildingType = Field(default=BuildingType.OFFICE, description="Building type")
    include_confidence_intervals: bool = Field(default=True, description="Include confidence intervals")

class ForecastResponse(BaseModel):
    """Response schema for energy consumption forecast"""
    forecast: List[EnergyDataPoint] = Field(..., description="Predicted energy consumption")
    confidence_intervals: Optional[Dict[str, List[float]]] = Field(None, description="Confidence intervals")
    model_performance: Optional[Dict[str, float]] = Field(None, description="Model performance metrics")
    total_forecast_consumption: float = Field(..., description="Total forecasted consumption")
    average_hourly_consumption: float = Field(..., description="Average hourly consumption")
    peak_consumption_hour: int = Field(..., description="Hour with peak consumption")
    model_type: str = Field(..., description="Model used for forecasting")
    forecast_timestamp: str = Field(..., description="Timestamp of forecast")

class BatchForecastRequest(BaseModel):
    """Request schema for batch energy forecasting"""
    historical_data_list: List[List[EnergyDataPoint]] = Field(..., description="List of historical data for multiple buildings")
    forecast_hours: int = Field(default=24, ge=1, le=168, description="Number of hours to forecast")
    model_type: ModelType = Field(default=ModelType.XGBOOST, description="Model type to use")
    building_types: List[BuildingType] = Field(..., description="Building types for each dataset")

class BatchForecastResponse(BaseModel):
    """Response schema for batch energy forecasting"""
    forecasts: List[ForecastResponse] = Field(..., description="List of forecasts for each building")
    summary: Dict[str, float] = Field(..., description="Summary statistics across all buildings")
    processing_time_seconds: float = Field(..., description="Processing time in seconds")

class ModelInfo(BaseModel):
    """Schema for model information"""
    model_type: str = Field(..., description="Type of model")
    training_date: str = Field(..., description="Date when model was trained")
    mae: float = Field(..., ge=0, description="Mean Absolute Error")
    rmse: float = Field(..., ge=0, description="Root Mean Square Error")
    mape: float = Field(..., ge=0, description="Mean Absolute Percentage Error")
    training_samples: int = Field(..., ge=0, description="Number of training samples")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance scores")

class ModelComparisonRequest(BaseModel):
    """Request schema for model comparison"""
    historical_data: List[EnergyDataPoint] = Field(..., description="Historical energy consumption data")
    test_hours: int = Field(default=24, ge=1, le=168, description="Number of hours for testing")
    building_type: BuildingType = Field(default=BuildingType.OFFICE, description="Building type")

class ModelComparisonResponse(BaseModel):
    """Response schema for model comparison"""
    comparison_results: List[ModelInfo] = Field(..., description="Performance comparison of different models")
    best_model: str = Field(..., description="Best performing model")
    recommendation: str = Field(..., description="Recommendation for model selection")

class EnergyEfficiencyRequest(BaseModel):
    """Request schema for energy efficiency analysis"""
    historical_data: List[EnergyDataPoint] = Field(..., description="Historical energy consumption data")
    building_type: BuildingType = Field(default=BuildingType.OFFICE, description="Building type")
    baseline_period_days: int = Field(default=30, ge=7, le=365, description="Baseline period in days")

class EnergyEfficiencyResponse(BaseModel):
    """Response schema for energy efficiency analysis"""
    efficiency_score: float = Field(..., ge=0, le=100, description="Energy efficiency score (0-100)")
    baseline_consumption: float = Field(..., description="Baseline consumption in kWh")
    current_consumption: float = Field(..., description="Current consumption in kWh")
    savings_potential_kwh: float = Field(..., description="Potential savings in kWh")
    savings_potential_percent: float = Field(..., description="Potential savings percentage")
    recommendations: List[str] = Field(..., description="Energy efficiency recommendations")
    peak_hours: List[int] = Field(..., description="Peak consumption hours")
    off_peak_hours: List[int] = Field(..., description="Off-peak consumption hours")

class AnomalyDetectionRequest(BaseModel):
    """Request schema for energy consumption anomaly detection"""
    historical_data: List[EnergyDataPoint] = Field(..., description="Historical energy consumption data")
    sensitivity: float = Field(default=0.05, ge=0.01, le=0.5, description="Anomaly detection sensitivity")
    building_type: BuildingType = Field(default=BuildingType.OFFICE, description="Building type")

class AnomalyDetectionResponse(BaseModel):
    """Response schema for energy consumption anomaly detection"""
    anomalies: List[EnergyDataPoint] = Field(..., description="Detected anomalies")
    anomaly_count: int = Field(..., description="Number of anomalies detected")
    anomaly_percentage: float = Field(..., description="Percentage of anomalous data points")
    normal_consumption_range: Dict[str, float] = Field(..., description="Normal consumption range")
    recommendations: List[str] = Field(..., description="Recommendations for anomaly handling")

class HealthResponse(BaseModel):
    """Response schema for health check"""
    status: str = Field(..., description="System status")
    available_models: List[str] = Field(..., description="List of available models")
    loaded_models: List[str] = Field(..., description="List of loaded models")
    timestamp: str = Field(..., description="Timestamp of health check")