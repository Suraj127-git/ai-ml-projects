"""
Credit Card Fraud Detection API Package

This package provides a FastAPI-based REST API for detecting credit card fraud
using machine learning models. It includes:

- Fraud detection model implementation
- API endpoints for prediction
- Data validation schemas
- Model training utilities
"""

__version__ = "1.0.0"
__author__ = "AI ML Projects"
__description__ = "Credit Card Fraud Detection API"

from .model import FraudDetectionModel
from .schemas import TransactionInput, FraudPrediction, ModelInfo

__all__ = [
    "FraudDetectionModel",
    "TransactionInput", 
    "FraudPrediction",
    "ModelInfo"
]