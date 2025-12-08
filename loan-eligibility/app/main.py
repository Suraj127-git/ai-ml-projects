from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import os
import sys

# Add the app directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from schemas import ApplicantData, LoanPrediction, ModelInfo, HealthResponse
from model import LoanEligibilityModel

# Initialize FastAPI app
app = FastAPI(
    title="Loan Eligibility Predictor API",
    description="API for predicting loan eligibility using machine learning models",
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
model = LoanEligibilityModel()
MODEL_PATH = "loan_eligibility_model.joblib"

# Load model on startup
@app.on_event("startup")
async def startup_event():
    try:
        if os.path.exists(MODEL_PATH):
            model.load_model(MODEL_PATH)
            print("Model loaded successfully!")
        else:
            print("No pre-trained model found. Please train the model first.")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="loan-eligibility-predictor",
        model_loaded=model.model is not None
    )

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information and performance metrics"""
    if not model.model_info:
        raise HTTPException(status_code=404, detail="Model not trained yet")
    
    return ModelInfo(**model.model_info)

@app.post("/predict", response_model=LoanPrediction)
async def predict_loan_eligibility(applicant_data: ApplicantData):
    """Predict loan eligibility for an applicant"""
    if model.model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not available. Please train the model first using /train endpoint."
        )
    
    try:
        # Convert Pydantic model to dictionary
        applicant_dict = applicant_data.dict()
        
        # Map field names to match model expectations
        mapping = {
            'applicant_income': 'ApplicantIncome',
            'coapplicant_income': 'CoapplicantIncome',
            'loan_amount': 'LoanAmount',
            'loan_amount_term': 'Loan_Amount_Term',
            'credit_history': 'Credit_History',
            'property_area': 'Property_Area',
            'self_employed': 'Self_Employed'
        }
        
        # Create properly formatted input
        formatted_input = {}
        for key, value in applicant_dict.items():
            if key in mapping:
                formatted_input[mapping[key]] = value
            else:
                formatted_input[key.capitalize()] = value
        
        # Make prediction
        prediction = model.predict(formatted_input)
        
        return LoanPrediction(**prediction)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/train")
async def train_model(model_type: str = "gradient_boosting"):
    """Train the loan eligibility model"""
    try:
        # Validate model type
        if model_type not in ["logistic_regression", "gradient_boosting"]:
            raise HTTPException(
                status_code=400, 
                detail="Invalid model type. Choose 'logistic_regression' or 'gradient_boosting'."
            )
        
        # Train model
        metrics = model.train_model(model_type)
        
        # Save model
        model.save_model(MODEL_PATH)
        
        return {
            "message": f"Model trained successfully using {model_type}",
            "metrics": metrics,
            "model_info": model.model_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@app.post("/batch_predict", response_model=List[LoanPrediction])
async def batch_predict(applicants: List[ApplicantData]):
    """Predict loan eligibility for multiple applicants"""
    if model.model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not available. Please train the model first."
        )
    
    predictions = []
    for applicant in applicants:
        try:
            prediction = await predict_loan_eligibility(applicant)
            predictions.append(prediction)
        except Exception as e:
            # Return a default prediction for failed cases
            predictions.append(LoanPrediction(
                eligibility="Rejected",
                probability=0.0,
                confidence="Low",
                key_factors=["Prediction failed"],
                risk_score=100.0
            ))
    
    return predictions

@app.get("/feature_importance")
async def get_feature_importance():
    """Get feature importance from the trained model"""
    if model.model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    if hasattr(model.model, 'feature_importances_'):
        importance_dict = dict(zip(model.feature_names, model.model.feature_importances_))
        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        return {
            "feature_importance": sorted_importance,
            "top_features": list(sorted_importance.keys())[:10]
        }
    else:
        return {"message": "Feature importance not available for this model type"}

@app.delete("/model")
async def delete_model():
    """Delete the trained model"""
    try:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
            model.model = None
            model.model_info = {}
            return {"message": "Model deleted successfully"}
        else:
            return {"message": "No model file found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting model: {str(e)}")

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return HTTPException(status_code=400, detail=str(exc))

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return HTTPException(status_code=500, detail=f"Internal server error: {str(exc)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)