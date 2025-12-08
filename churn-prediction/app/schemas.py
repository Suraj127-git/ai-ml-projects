from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

class CustomerData(BaseModel):
    """Schema for customer churn prediction input data"""
    
    # Customer demographics
    customer_id: str = Field(..., description="Unique customer identifier")
    gender: str = Field(..., description="Customer gender (Male/Female)")
    senior_citizen: int = Field(..., ge=0, le=1, description="Senior citizen status (0/1)")
    partner: str = Field(..., description="Has partner (Yes/No)")
    dependents: str = Field(..., description="Has dependents (Yes/No)")
    
    # Service information
    tenure: int = Field(..., ge=0, le=120, description="Number of months the customer has stayed with the company")
    phone_service: str = Field(..., description="Has phone service (Yes/No)")
    multiple_lines: str = Field(..., description="Has multiple lines (Yes/No/No phone service)")
    internet_service: str = Field(..., description="Internet service type (DSL/Fiber optic/No)")
    online_security: str = Field(..., description="Has online security (Yes/No/No internet service)")
    online_backup: str = Field(..., description="Has online backup (Yes/No/No internet service)")
    device_protection: str = Field(..., description="Has device protection (Yes/No/No internet service)")
    tech_support: str = Field(..., description="Has tech support (Yes/No/No internet service)")
    streaming_tv: str = Field(..., description="Has streaming TV (Yes/No/No internet service)")
    streaming_movies: str = Field(..., description="Has streaming movies (Yes/No/No internet service)")
    
    # Account information
    contract: str = Field(..., description="Contract type (Month-to-month/One year/Two year)")
    paperless_billing: str = Field(..., description="Has paperless billing (Yes/No)")
    payment_method: str = Field(..., description="Payment method (Electronic check/Mailed check/Bank transfer/Credit card)")
    
    # Financial information
    monthly_charges: float = Field(..., ge=0, description="Monthly charges amount")
    total_charges: float = Field(..., ge=0, description="Total charges amount")

class ChurnPrediction(BaseModel):
    """Schema for churn prediction output"""
    
    customer_id: str = Field(..., description="Customer identifier")
    churn_probability: float = Field(..., ge=0, le=1, description="Probability of customer churning")
    churn_prediction: str = Field(..., description="Churn prediction (Yes/No)")
    confidence: str = Field(..., description="Confidence level (High/Medium/Low)")
    risk_score: float = Field(..., ge=0, le=100, description="Risk score (0-100, higher is riskier)")
    key_factors: List[str] = Field(..., description="Key factors influencing the prediction")
    shap_explanation: Optional[Dict] = Field(None, description="SHAP explanation for the prediction")

class ModelInfo(BaseModel):
    """Schema for model information"""
    
    model_type: str = Field(..., description="Type of model used")
    training_date: str = Field(..., description="Date when model was trained")
    accuracy: float = Field(..., description="Model accuracy on test set")
    precision: float = Field(..., description="Model precision on test set")
    recall: float = Field(..., description="Model recall on test set")
    f1_score: float = Field(..., description="Model F1-score on test set")
    auc_roc: float = Field(..., description="Model AUC-ROC score")
    training_samples: int = Field(..., description="Number of training samples")
    feature_count: int = Field(..., description="Number of features used")

class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction request"""
    
    customers: List[CustomerData] = Field(..., description="List of customer data for batch prediction")
    include_shap: bool = Field(False, description="Whether to include SHAP explanations")

class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response"""
    
    predictions: List[ChurnPrediction] = Field(..., description="List of churn predictions")
    summary: Dict = Field(..., description="Summary statistics of batch predictions")

class HealthResponse(BaseModel):
    """Schema for health check response"""
    
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Current timestamp")

class FeatureImportance(BaseModel):
    """Schema for feature importance information"""
    
    feature_name: str = Field(..., description="Feature name")
    importance_score: float = Field(..., description="Importance score")
    description: Optional[str] = Field(None, description="Feature description")

class ModelPerformance(BaseModel):
    """Schema for model performance metrics"""
    
    metric_name: str = Field(..., description="Performance metric name")
    value: float = Field(..., description="Metric value")
    threshold: Optional[float] = Field(None, description="Acceptable threshold for the metric")