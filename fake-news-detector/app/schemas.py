from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum

class ModelType(str, Enum):
    TFIDF_LOGISTIC = "tfidf_logistic"
    BERT = "bert"
    ROBERTA = "roberta"

class FakeNewsRequest(BaseModel):
    text: str = Field(..., description="News text to analyze for fake news detection")
    title: Optional[str] = Field(None, description="News title (optional)")
    model_type: ModelType = Field(default=ModelType.TFIDF_LOGISTIC, description="Model to use for prediction")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Scientists discover that drinking water can make you live forever. This breakthrough changes everything we know about aging.",
                "title": "Amazing Water Discovery",
                "model_type": "tfidf_logistic"
            }
        }

class FakeNewsResponse(BaseModel):
    text: str = Field(..., description="Original text analyzed")
    title: Optional[str] = Field(None, description="Title if provided")
    prediction: str = Field(..., description="Prediction: 'FAKE' or 'REAL'")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")
    model_type: ModelType = Field(..., description="Model used for prediction")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Scientists discover that drinking water can make you live forever...",
                "title": "Amazing Water Discovery",
                "prediction": "FAKE",
                "confidence": 0.92,
                "model_type": "tfidf_logistic",
                "probabilities": {"FAKE": 0.92, "REAL": 0.08}
            }
        }

class BatchFakeNewsRequest(BaseModel):
    texts: List[FakeNewsRequest] = Field(..., description="List of news texts to analyze")
    model_type: ModelType = Field(default=ModelType.TFIDF_LOGISTIC, description="Model to use for predictions")

class BatchFakeNewsResponse(BaseModel):
    predictions: List[FakeNewsResponse] = Field(..., description="List of predictions")
    model_type: ModelType = Field(..., description="Model used for predictions")
    total_processed: int = Field(..., description="Total number of texts processed")

class ModelInfo(BaseModel):
    model_type: ModelType = Field(..., description="Model type")
    description: str = Field(..., description="Model description")
    accuracy: Optional[float] = Field(None, description="Model accuracy on test set")
    features: List[str] = Field(..., description="Model features")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status")
    models_loaded: List[ModelType] = Field(..., description="List of loaded models")
    available_models: List[ModelType] = Field(..., description="List of available models")