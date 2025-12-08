from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import Dict, Any, Optional
import uvicorn
import os

from .model import CLVModel
from .schemas import (
    CustomerData, CLVPrediction, ModelInfo, TrainingResponse, 
    HealthResponse, ModelName
)

# Initialize FastAPI app
app = FastAPI(
    title="Customer Lifetime Value (CLV) Predictor API",
    description="API for predicting customer lifetime value using XGBoost and BG-NBD models",
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
clv_model = CLVModel()

# Model storage
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    """Load models on startup if they exist"""
    xgboost_path = os.path.join(MODEL_DIR, "clv_xgboost_model.pkl")
    bg_nbd_path = os.path.join(MODEL_DIR, "clv_bg_nbd_model.pkl")
    
    try:
        if os.path.exists(xgboost_path):
            clv_model.load_model(xgboost_path)
            print("Loaded XGBoost model successfully")
    except Exception as e:
        print(f"Could not load XGBoost model: {e}")
    
    try:
        if os.path.exists(bg_nbd_path):
            clv_model.load_model(bg_nbd_path)
            print("Loaded BG-NBD model successfully")
    except Exception as e:
        print(f"Could not load BG-NBD model: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Customer Lifetime Value (CLV) Predictor API",
        "version": "1.0.0",
        "available_models": ["xgboost", "bg_nbd"],
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_info = clv_model.get_model_info()
    loaded_models = sum(model_info['models_trained'].values())
    
    return HealthResponse(
        status="healthy",
        available_models=["xgboost", "bg_nbd"],
        loaded_models=loaded_models,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=CLVPrediction)
async def predict_clv(customer_data: CustomerData, model_type: ModelName = ModelName.XGBOOST):
    """
    Predict Customer Lifetime Value
    
    Args:
        customer_data: Customer transaction data
        model_type: Model to use for prediction (xgboost or bg_nbd)
    """
    try:
        # Convert customer data to dict
        customer_dict = customer_data.dict()
        
        # Make prediction based on model type
        if model_type == ModelName.XGBOOST:
            if not clv_model.get_model_info()['models_trained']['xgboost']:
                raise HTTPException(
                    status_code=400, 
                    detail="XGBoost model not trained. Please train the model first."
                )
            prediction_result = clv_model.predict_clv_xgboost(customer_dict)
            
        elif model_type == ModelName.BG_NBD:
            if not clv_model.get_model_info()['models_trained']['bg_nbd']:
                raise HTTPException(
                    status_code=400, 
                    detail="BG-NBD model not trained. Please train the model first."
                )
            prediction_result = clv_model.predict_clv_bg_nbd(customer_dict)
        
        # Create response
        return CLVPrediction(
            customer_id=customer_data.customer_id,
            predicted_clv=prediction_result['predicted_clv'],
            model_used=prediction_result['model_used'],
            confidence_score=prediction_result.get('confidence_score'),
            prediction_date=datetime.now().date(),
            expected_purchases_next_30_days=prediction_result.get('expected_purchases_next_30_days'),
            expected_purchases_next_90_days=prediction_result.get('expected_purchases_next_90_days'),
            expected_purchases_next_365_days=prediction_result.get('expected_purchases_next_365_days')
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
async def predict_clv_batch(customers: list[CustomerData], model_type: ModelName = ModelName.XGBOOST):
    """
    Predict CLV for multiple customers in batch
    
    Args:
        customers: List of customer data
        model_type: Model to use for prediction
    """
    try:
        predictions = []
        
        for customer_data in customers:
            customer_dict = customer_data.dict()
            
            if model_type == ModelName.XGBOOST:
                prediction_result = clv_model.predict_clv_xgboost(customer_dict)
            elif model_type == ModelName.BG_NBD:
                prediction_result = clv_model.predict_clv_bg_nbd(customer_dict)
            
            prediction = CLVPrediction(
                customer_id=customer_data.customer_id,
                predicted_clv=prediction_result['predicted_clv'],
                model_used=prediction_result['model_used'],
                confidence_score=prediction_result.get('confidence_score'),
                prediction_date=datetime.now().date(),
                expected_purchases_next_30_days=prediction_result.get('expected_purchases_next_30_days'),
                expected_purchases_next_90_days=prediction_result.get('expected_purchases_next_90_days'),
                expected_purchases_next_365_days=prediction_result.get('expected_purchases_next_365_days')
            )
            predictions.append(prediction)
        
        return {
            "predictions": predictions,
            "model_type": model_type,
            "total_predictions": len(predictions),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Batch prediction error: {str(e)}')

@app.post("/train/{model_type}", response_model=TrainingResponse)
async def train_model(model_type: ModelName, n_samples: int = 1000):
    """
    Train a CLV model
    
    Args:
        model_type: Type of model to train (xgboost or bg_nbd)
        n_samples: Number of synthetic samples to generate for training
    """
    try:
        print(f"Training {model_type} model with {n_samples} samples...")
        
        # Generate synthetic training data
        df = clv_model.generate_synthetic_customer_data(n_samples)
        
        # Train model
        if model_type == ModelName.XGBOOST:
            performance = clv_model.train_xgboost_model(df)
            model_file = "clv_xgboost_model.pkl"
            
        elif model_type == ModelName.BG_NBD:
            performance = clv_model.train_bg_nbd_model(df)
            model_file = "clv_bg_nbd_model.pkl"
        
        # Save model
        model_path = os.path.join(MODEL_DIR, model_file)
        clv_model.save_model(model_path, model_type)
        
        return TrainingResponse(
            message=f"{model_type} model trained successfully",
            model_type=model_type,
            training_samples=n_samples,
            model_performance=performance
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about available models"""
    try:
        model_info = clv_model.get_model_info()
        
        return ModelInfo(
            model_name="CLV Predictor",
            model_type="Ensemble (XGBoost + BG-NBD)",
            features=[
                "recency", "frequency", "monetary_value", "tenure",
                "avg_order_value", "days_between_purchases", "total_orders",
                "purchase_frequency", "value_per_day", "recency_score"
            ],
            training_date=datetime.now().date(),
            model_performance=model_info['model_performance'],
            version="1.0.0"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@app.get("/model/performance")
async def get_model_performance():
    """Get detailed model performance metrics"""
    try:
        model_info = clv_model.get_model_info()
        return {
            "model_performance": model_info['model_performance'],
            "models_available": model_info['xgboost_available'] and model_info['lifetimes_available'],
            "models_trained": model_info['models_trained'],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model performance: {str(e)}")

@app.post("/generate-sample-data")
async def generate_sample_data(n_customers: int = 100):
    """
    Generate sample customer data for testing
    
    Args:
        n_customers: Number of sample customers to generate
    """
    try:
        df = clv_model.generate_synthetic_customer_data(n_customers)
        
        # Convert to list of CustomerData objects
        customers = []
        for _, row in df.iterrows():
            customer = CustomerData(
                customer_id=row['customer_id'],
                recency=int(row['recency']),
                frequency=int(row['frequency']),
                monetary_value=float(row['monetary_value']),
                tenure=int(row['tenure']),
                avg_order_value=float(row['avg_order_value']),
                days_between_purchases=float(row['days_between_purchases']),
                total_orders=int(row['total_orders']),
                age=int(row['age']) if pd.notna(row['age']) else None,
                gender=row['gender'] if pd.notna(row['gender']) else None,
                country=row['country'] if pd.notna(row['country']) else None
            )
            customers.append(customer)
        
        return {
            "sample_customers": customers,
            "total_generated": len(customers),
            "summary_stats": {
                "avg_clv": float(df['clv'].mean()),
                "avg_frequency": float(df['frequency'].mean()),
                "avg_monetary_value": float(df['monetary_value'].mean()),
                "avg_tenure": float(df['tenure'].mean())
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating sample data: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)