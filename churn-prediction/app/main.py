from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import os
import sys
import pandas as pd
from datetime import datetime

# Add the app directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from schemas import CustomerData, ChurnPrediction, ModelInfo, HealthResponse, BatchPredictionRequest, BatchPredictionResponse, FeatureImportance
from model import ChurnPredictionModel

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using XGBoost and SHAP interpretability",
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
model = ChurnPredictionModel()
MODEL_PATH = "churn_prediction_model.joblib"

# Load model on startup
@app.on_event("startup")
async def startup_event():
    try:
        if os.path.exists(MODEL_PATH):
            model.load_model(MODEL_PATH)
            print("Churn prediction model loaded successfully!")
        else:
            print("No pre-trained model found. Please train the model first.")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="churn-prediction-api",
        model_loaded=model.model is not None
    )

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information and performance metrics"""
    if not model.model_info:
        raise HTTPException(status_code=404, detail="Model not trained yet")
    
    return ModelInfo(**model.model_info)

@app.post("/predict", response_model=ChurnPrediction)
async def predict_churn(customer_data: CustomerData, include_shap: bool = False):
    """Predict churn probability for a single customer"""
    if model.model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not available. Please train the model first using /train endpoint."
        )
    
    try:
        # Convert Pydantic model to dictionary
        customer_dict = customer_data.dict()
        
        # Make prediction
        prediction = model.predict(customer_dict, include_shap=include_shap)
        
        return ChurnPrediction(**prediction)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_churn(batch_request: BatchPredictionRequest):
    """Predict churn probability for multiple customers"""
    if model.model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not available. Please train the model first."
        )
    
    try:
        predictions = []
        churn_count = 0
        total_probability = 0
        
        for customer in batch_request.customers:
            try:
                prediction = model.predict(customer.dict(), include_shap=batch_request.include_shap)
                predictions.append(ChurnPrediction(**prediction))
                
                if prediction['churn_prediction'] == 'Yes':
                    churn_count += 1
                total_probability += prediction['churn_probability']
                
            except Exception as e:
                # Return a default prediction for failed cases
                failed_prediction = ChurnPrediction(
                    customer_id=customer.customer_id,
                    churn_probability=0.0,
                    churn_prediction="No",
                    confidence="Low",
                    risk_score=0.0,
                    key_factors=["Prediction failed"]
                )
                predictions.append(failed_prediction)
        
        # Calculate summary statistics
        summary = {
            "total_customers": len(predictions),
            "churn_predictions": churn_count,
            "retention_predictions": len(predictions) - churn_count,
            "average_churn_probability": total_probability / len(predictions) if predictions else 0,
            "churn_rate": churn_count / len(predictions) if predictions else 0
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Batch prediction error: {str(e)}')

@app.post("/train")
async def train_model():
    """Train the churn prediction model"""
    try:
        # Train model
        metrics = model.train_model('xgboost')
        
        # Save model
        model.save_model(MODEL_PATH)
        
        return {
            "message": "Churn prediction model trained successfully",
            "metrics": metrics,
            "model_info": model.model_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@app.get("/model/feature_importance")
async def get_feature_importance():
    """Get feature importance from the trained model"""
    if model.model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        importance_data = model.get_feature_importance()
        
        if 'feature_importance' in importance_data:
            # Convert to list of FeatureImportance objects
            feature_importances = []
            for feature, importance in importance_data['feature_importance'].items():
                feature_importances.append(FeatureImportance(
                    feature_name=feature,
                    importance_score=float(importance),
                    description=f"Importance of {feature} for churn prediction"
                ))
            
            return {
                "feature_importances": feature_importances,
                "top_features": importance_data.get('top_features', [])
            }
        else:
            return importance_data
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting feature importance: {str(e)}")

@app.get("/model/shap_analysis/{customer_id}")
async def get_shap_analysis(customer_id: str, 
                          gender: str = "Male",
                          senior_citizen: int = 0,
                          partner: str = "No", 
                          dependents: str = "No",
                          tenure: int = 12,
                          phone_service: str = "Yes",
                          multiple_lines: str = "No",
                          internet_service: str = "Fiber optic",
                          online_security: str = "No",
                          online_backup: str = "No", 
                          device_protection: str = "No",
                          tech_support: str = "No",
                          streaming_tv: str = "No",
                          streaming_movies: str = "No",
                          contract: str = "Month-to-month",
                          paperless_billing: str = "Yes",
                          payment_method: str = "Electronic check",
                          monthly_charges: float = 70.0,
                          total_charges: float = 840.0):
    """Get SHAP analysis for a customer profile"""
    if model.model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    if model.shap_explainer is None:
        raise HTTPException(status_code=503, detail="SHAP explainer not available")
    
    try:
        # Create customer data
        customer_data = CustomerData(
            customer_id=customer_id,
            gender=gender,
            senior_citizen=senior_citizen,
            partner=partner,
            dependents=dependents,
            tenure=tenure,
            phone_service=phone_service,
            multiple_lines=multiple_lines,
            internet_service=internet_service,
            online_security=online_security,
            online_backup=online_backup,
            device_protection=device_protection,
            tech_support=tech_support,
            streaming_tv=streaming_tv,
            streaming_movies=streaming_movies,
            contract=contract,
            paperless_billing=paperless_billing,
            payment_method=payment_method,
            monthly_charges=monthly_charges,
            total_charges=total_charges
        )
        
        # Get prediction with SHAP explanation
        prediction = await predict_churn(customer_data, include_shap=True)
        
        return prediction
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'SHAP analysis error: {str(e)}')

@app.get("/analytics/summary")
async def get_analytics_summary():
    """Get analytics summary for the churn prediction model"""
    if model.model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        # Generate some sample data for analytics
        sample_data = model.generate_synthetic_data(1000)
        
        # Calculate summary statistics
        summary = {
            "sample_size": len(sample_data),
            "churn_rate": (sample_data['Churn'] == 'Yes').mean(),
            "avg_tenure": sample_data['tenure'].mean(),
            "avg_monthly_charges": sample_data['MonthlyCharges'].mean(),
            "avg_total_charges": sample_data['TotalCharges'].mean(),
            "most_common_contract": sample_data['Contract'].mode()[0],
            "most_common_payment_method": sample_data['PaymentMethod'].mode()[0],
            "senior_citizen_percentage": (sample_data['SeniorCitizen'] == 1).mean(),
            "partner_percentage": (sample_data['Partner'] == 'Yes').mean(),
            "dependents_percentage": (sample_data['Dependents'] == 'Yes').mean(),
            "fiber_optic_percentage": (sample_data['InternetService'] == 'Fiber optic').mean(),
            "month_to_month_percentage": (sample_data['Contract'] == 'Month-to-month').mean()
        }
        
        return summary
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics error: {str(e)}")

@app.delete("/model")
async def delete_model():
    """Delete the trained model"""
    try:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
            model.model = None
            model.model_info = {}
            model.shap_explainer = None
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
    uvicorn.run(app, host="0.0.0.0", port=8003)