from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class ApplicantData(BaseModel):
    """Input schema for loan eligibility prediction"""
    gender: Literal["Male", "Female"] = Field(..., description="Applicant gender")
    married: Literal["Yes", "No"] = Field(..., description="Marital status")
    dependents: int = Field(..., ge=0, le=10, description="Number of dependents")
    education: Literal["Graduate", "Not Graduate"] = Field(..., description="Education level")
    self_employed: Literal["Yes", "No"] = Field(..., description="Self employment status")
    applicant_income: float = Field(..., ge=0, description="Applicant monthly income")
    coapplicant_income: float = Field(..., ge=0, description="Co-applicant monthly income")
    loan_amount: float = Field(..., ge=0, description="Loan amount requested")
    loan_amount_term: int = Field(..., ge=12, le=480, description="Loan term in months")
    credit_history: Literal["Yes", "No"] = Field(..., description="Credit history (1=Good, 0=Bad)")
    property_area: Literal["Urban", "Semiurban", "Rural"] = Field(..., description="Property location")
    
class LoanPrediction(BaseModel):
    """Output schema for loan eligibility prediction"""
    eligibility: Literal["Approved", "Rejected"] = Field(..., description="Loan eligibility status")
    probability: float = Field(..., description="Probability of approval")
    confidence: str = Field(..., description="Confidence level (High/Medium/Low)")
    key_factors: List[str] = Field(..., description="Key factors influencing the decision")
    risk_score: float = Field(..., description="Risk score (0-100, higher is riskier)")
    
class ModelInfo(BaseModel):
    """Schema for model information"""
    model_type: str
    training_date: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    training_samples: int