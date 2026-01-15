"""
Pydantic schemas for Predictive Maintenance System API
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime, date
from enum import Enum

class EquipmentType(str, Enum):
    """Equipment types for maintenance prediction"""
    PUMP = "Pump"
    MOTOR = "Motor"
    COMPRESSOR = "Compressor"
    FAN = "Fan"
    BEARING = "Bearing"
    GEARBOX = "Gearbox"

class MaintenanceType(str, Enum):
    """Types of maintenance"""
    PREVENTIVE = "Preventive"
    CORRECTIVE = "Corrective"
    PREDICTIVE = "Predictive"
    CONDITION_BASED = "Condition-Based"

class EnvironmentalCondition(str, Enum):
    """Environmental conditions"""
    NORMAL = "Normal"
    HARSH = "Harsh"
    CORROSIVE = "Corrosive"
    HIGH_TEMPERATURE = "High Temperature"
    HIGH_HUMIDITY = "High Humidity"
    DUSTY = "Dusty"

class RiskLevel(str, Enum):
    """Risk levels for equipment failure"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class EquipmentData(BaseModel):
    """Input schema for equipment data"""
    equipment_id: str = Field(..., description="Unique equipment identifier")
    equipment_type: EquipmentType = Field(..., description="Type of equipment")
    operating_hours: float = Field(..., ge=0, description="Total operating hours")
    temperature: float = Field(..., description="Current temperature in Celsius")
    vibration: float = Field(..., ge=0, description="Vibration level")
    pressure: float = Field(..., ge=0, description="Pressure level")
    days_since_maintenance: int = Field(..., ge=0, description="Days since last maintenance")
    last_maintenance_type: MaintenanceType = Field(..., description="Type of last maintenance")
    maintenance_frequency: int = Field(..., ge=1, description="Maintenance frequency in days")
    environmental_conditions: EnvironmentalCondition = Field(..., description="Environmental conditions")
    load_factor: float = Field(..., ge=0, le=1, description="Load factor (0-1)")
    age_years: float = Field(..., ge=0, description="Equipment age in years")

class PredictionRequest(BaseModel):
    """Request schema for failure prediction"""
    equipment_data: EquipmentData = Field(..., description="Equipment data for prediction")
    model_type: str = Field(default="random_forest", description="Model type: random_forest, gradient_boosting, or logistic_regression")

class MaintenanceTask(BaseModel):
    """Schema for maintenance task"""
    task_type: str = Field(..., description="Type of maintenance task")
    description: str = Field(..., description="Task description")
    priority: str = Field(..., description="Task priority")
    estimated_duration_hours: float = Field(..., description="Estimated duration in hours")
    estimated_cost: float = Field(..., description="Estimated cost in currency units")

class MaintenanceSchedule(BaseModel):
    """Schema for maintenance schedule"""
    recommended_maintenance_date: date = Field(..., description="Recommended maintenance date")
    urgency_level: str = Field(..., description="Urgency level")
    maintenance_tasks: List[MaintenanceTask] = Field(..., description="List of recommended maintenance tasks")
    total_estimated_cost: float = Field(..., description="Total estimated cost")
    total_estimated_duration: float = Field(..., description="Total estimated duration in hours")

class PredictionResponse(BaseModel):
    """Response schema for failure prediction"""
    equipment_id: str = Field(..., description="Equipment identifier")
    failure_probability: float = Field(..., ge=0, le=1, description="Probability of failure (0-1)")
    risk_level: RiskLevel = Field(..., description="Risk level")
    maintenance_schedule: MaintenanceSchedule = Field(..., description="Recommended maintenance schedule")
    recommended_actions: List[str] = Field(..., description="List of recommended actions")
    model_type: str = Field(..., description="Model used for prediction")
    prediction_timestamp: str = Field(..., description="Timestamp of prediction")

class ModelInfo(BaseModel):
    """Schema for model information"""
    model_type: str = Field(..., description="Type of model")
    training_date: str = Field(..., description="Date when model was trained")
    accuracy: float = Field(..., ge=0, le=1, description="Model accuracy")
    precision: float = Field(..., ge=0, le=1, description="Model precision")
    recall: float = Field(..., ge=0, le=1, description="Model recall")
    f1_score: float = Field(..., ge=0, le=1, description="Model F1 score")
    training_samples: int = Field(..., description="Number of training samples")

class HealthResponse(BaseModel):
    """Response schema for health check"""
    status: str = Field(..., description="System status")
    available_models: List[str] = Field(..., description="List of available models")
    loaded_models: List[str] = Field(..., description="List of loaded models")
    timestamp: str = Field(..., description="Timestamp of health check")

class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions"""
    equipment_data_list: List[EquipmentData] = Field(..., description="List of equipment data")
    model_type: str = Field(default="random_forest", description="Model type to use")

class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions"""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    summary: Dict[str, int] = Field(..., description="Summary statistics")
    processing_time_seconds: float = Field(..., description="Processing time in seconds")