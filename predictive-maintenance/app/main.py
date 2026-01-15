"""
Predictive Maintenance System API
FastAPI-based API for equipment failure prediction and maintenance scheduling
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

from .model import PredictiveMaintenanceModel
from .schemas import (
    EquipmentData, PredictionRequest, PredictionResponse,
    MaintenanceSchedule, ModelInfo, HealthResponse
)

# Initialize FastAPI app
app = FastAPI(
    title="Predictive Maintenance System",
    description="API for equipment failure prediction and maintenance scheduling",
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
    'random_forest': None,
    'gradient_boosting': None,
    'logistic_regression': None
}

def load_models():
    """Load pre-trained models"""
    model_files = {
        'random_forest': 'maintenance_rf_model.pkl',
        'gradient_boosting': 'maintenance_gb_model.pkl',
        'logistic_regression': 'maintenance_lr_model.pkl'
    }
    
    for model_type, filename in model_files.items():
        try:
            if os.path.exists(filename):
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
        "message": "Predictive Maintenance System API",
        "version": "1.0.0",
        "status": "active",
        "available_models": list(models.keys())
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_failure(request: PredictionRequest):
    """
    Predict equipment failure probability
    
    Args:
        request: Equipment data and prediction parameters
        
    Returns:
        Failure prediction with probability and maintenance recommendations
    """
    try:
        model_type = request.model_type.lower()
        
        if model_type not in models:
            raise HTTPException(status_code=400, detail=f'Model {model_type} not available')
        
        if not models[model_type]:
            raise HTTPException(status_code=503, detail=f'Model {model_type} not loaded')
        
        # Create equipment data dictionary
        equipment_data = {
            'equipment_type': request.equipment_data.equipment_type,
            'operating_hours': request.equipment_data.operating_hours,
            'temperature': request.equipment_data.temperature,
            'vibration': request.equipment_data.vibration,
            'pressure': request.equipment_data.pressure,
            'days_since_maintenance': request.equipment_data.days_since_maintenance,
            'last_maintenance_type': request.equipment_data.last_maintenance_type,
            'maintenance_frequency': request.equipment_data.maintenance_frequency,
            'environmental_conditions': request.equipment_data.environmental_conditions,
            'load_factor': request.equipment_data.load_factor,
            'age_years': request.equipment_data.age_years
        }
        
        # Initialize model
        pm_model = PredictiveMaintenanceModel()
        
        # Predict failure probability
        failure_probability = pm_model.predict_failure_probability(equipment_data, model_type)
        
        # Generate maintenance schedule
        maintenance_schedule = pm_model.generate_maintenance_schedule(
            equipment_data, failure_probability
        )
        
        # Determine risk level
        risk_level = "low"
        if failure_probability > 0.7:
            risk_level = "high"
        elif failure_probability > 0.4:
            risk_level = "medium"
        
        return PredictionResponse(
            equipment_id=request.equipment_data.equipment_id,
            failure_probability=failure_probability,
            risk_level=risk_level,
            maintenance_schedule=maintenance_schedule,
            recommended_actions=pm_model.get_recommended_actions(failure_probability, equipment_data),
            model_type=model_type,
            prediction_timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Prediction error: {str(e)}')

@app.post("/predict/{model_type}", response_model=PredictionResponse)
async def predict_with_model(model_type: str, request: PredictionRequest):
    """
    Predict equipment failure using specific model
    
    Args:
        model_type: Type of model to use (random_forest, gradient_boosting, logistic_regression)
        request: Equipment data and prediction parameters
        
    Returns:
        Failure prediction with probability and maintenance recommendations
    """
    try:
        if model_type.lower() not in models:
            raise HTTPException(status_code=400, detail=f'Model {model_type} not available')
        
        # Update request with specific model type
        request.model_type = model_type
        return await predict_failure(request)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Prediction error: {str(e)}')

@app.get("/models")
async def get_available_models():
    """Get list of available prediction models"""
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
        accuracy=0.0,  # Placeholder - would come from actual evaluation
        precision=0.0,   # Placeholder - would come from actual evaluation
        recall=0.0,  # Placeholder - would come from actual evaluation
        f1_score=0.0,  # Placeholder - would come from actual evaluation
        training_samples=0  # Placeholder - would come from actual training
    )

@app.post("/generate-sample-data")
async def generate_sample_data(n_samples: int = 100):
    """
    Generate sample maintenance data for testing
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Sample equipment data
    """
    try:
        model = PredictiveMaintenanceModel()
        sample_df = model.generate_synthetic_maintenance_data(n_samples=n_samples)
        
        # Convert to response format
        sample_data = []
        for _, row in sample_df.iterrows():
            sample_data.append({
                "equipment_id": f"EQ{row['equipment_id']}",
                "equipment_type": row['equipment_type'],
                "operating_hours": row['operating_hours'],
                "temperature": row['temperature'],
                "vibration": row['vibration'],
                "pressure": row['pressure'],
                "days_since_maintenance": row['days_since_maintenance'],
                "last_maintenance_type": row['last_maintenance_type'],
                "maintenance_frequency": row['maintenance_frequency'],
                "environmental_conditions": row['environmental_conditions'],
                "load_factor": row['load_factor'],
                "age_years": row['age_years'],
                "will_fail": bool(row['will_fail'])
            })
        
        return {
            "sample_data": sample_data,
            "count": len(sample_data),
            "summary": {
                "total_samples": len(sample_data),
                "failure_rate": sample_df['will_fail'].mean(),
                "equipment_types": sample_df['equipment_type'].value_counts().to_dict()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Data generation error: {str(e)}')

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
async def train_model(model_type: str):
    """
    Train a predictive maintenance model (for development)
    
    Args:
        model_type: Type of model to train (random_forest, gradient_boosting, logistic_regression)
    """
    try:
        print(f"Training {model_type} model...")
        
        # Generate training data
        model = PredictiveMaintenanceModel()
        df = model.generate_synthetic_maintenance_data(n_samples=1000)
        
        # Train specific model
        if model_type == 'random_forest':
            model.train_random_forest_model(df)
            model.save_model('maintenance_rf_model.pkl', 'random_forest')
        elif model_type == 'gradient_boosting':
            model.train_gradient_boosting_model(df)
            model.save_model('maintenance_gb_model.pkl', 'gradient_boosting')
        elif model_type == 'logistic_regression':
            model.train_logistic_regression_model(df)
            model.save_model('maintenance_lr_model.pkl', 'logistic_regression')
        else:
            raise HTTPException(status_code=400, detail=f'Unknown model type: {model_type}')
        
        # Reload models
        load_models()
        
        return {
            "message": f"{model_type} model trained successfully",
            "model_type": model_type,
            "training_samples": len(df)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Training error: {str(e)}')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)