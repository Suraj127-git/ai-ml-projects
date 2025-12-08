from pydantic import BaseModel, Field
from typing import List, Optional

class TransactionInput(BaseModel):
    """Input schema for fraud detection"""
    transaction_amount: float = Field(..., description="Transaction amount")
    merchant_category: str = Field(..., description="Merchant category code")
    card_type: str = Field(..., description="Type of card (credit/debit)")
    transaction_type: str = Field(..., description="Type of transaction")
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour of transaction")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday)")
    customer_age: int = Field(..., ge=18, le=100, description="Customer age")
    account_balance: float = Field(..., description="Account balance before transaction")
    previous_transaction_amount: float = Field(..., description="Previous transaction amount")
    transaction_frequency_24h: int = Field(..., description="Number of transactions in last 24h")
    
class FraudPrediction(BaseModel):
    """Output schema for fraud prediction"""
    fraud_probability: float = Field(..., description="Probability of fraud (0-1)")
    is_fraud: bool = Field(..., description="Binary fraud prediction")
    confidence: str = Field(..., description="Confidence level (Low/Medium/High)")
    risk_factors: List[str] = Field(..., description="List of risk factors identified")
    
class ModelInfo(BaseModel):
    """Schema for model information"""
    model_type: str
    training_date: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float