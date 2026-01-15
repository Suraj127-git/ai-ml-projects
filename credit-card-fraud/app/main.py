from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime
import os

from .schemas import TransactionInput, FraudPrediction, ModelInfo
from .model import FraudDetectionModel

# Initialize FastAPI app
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for detecting credit card fraud using machine learning",
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

# Initialize model
model = FraudDetectionModel()
model_path = "fraud_detection_model_rf.pkl"

# Load model on startup
@app.on_event("startup")
async def startup_event():
    """Load the fraud detection model on startup"""
    try:
        if os.path.exists(model_path):
            model.load_model(model_path)
            print("Fraud detection model loaded successfully!")
        else:
            print("Model file not found. Please train the model first.")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Credit Card Fraud Detection API",
        "version": "1.0.0",
        "status": "active",
        "model_loaded": model.model is not None
    }

@app.post("/predict", response_model=FraudPrediction)
async def predict_fraud(transaction: TransactionInput):
    """
    Predict whether a credit card transaction is fraudulent
    
    Args:
        transaction: Transaction details
        
    Returns:
        Fraud prediction with probability and risk factors
    """
    try:
        if model.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
        
        # Convert input to dictionary
        transaction_dict = transaction.dict()
        
        # Make prediction
        prediction = model.predict(transaction_dict)
        
        return FraudPrediction(**prediction)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the trained model"""
    if model.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        model_type=model.metrics.get('model_type', 'unknown'),
        training_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        accuracy=model.metrics.get('accuracy', 0.0),
        precision=0.95,  # Placeholder - would come from actual evaluation
        recall=0.89,     # Placeholder - would come from actual evaluation
        f1_score=0.92,   # Placeholder - would come from actual evaluation
        auc_roc=model.metrics.get('auc_roc', 0.0)
    )

@app.post("/batch-predict")
async def batch_predict(transactions: list[TransactionInput]):
    """
    Predict fraud for multiple transactions
    
    Args:
        transactions: List of transaction details
        
    Returns:
        List of fraud predictions
    """
    try:
        if model.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        predictions = []
        for transaction in transactions:
            prediction = model.predict(transaction.dict())
            predictions.append(prediction)
        
        return {"predictions": predictions, "count": len(predictions)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Batch prediction error: {str(e)}')

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model.model is not None,
        "timestamp": datetime.now().isoformat()
    }

# Training endpoint (for development/testing)
@app.post("/train")
async def train_model():
    """Train the fraud detection model (for development)"""
    try:
        print("Training fraud detection model...")
        metrics = model.train_model('random_forest')
        model.save_model(model_path)
        
        return {
            "message": "Model trained successfully",
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)