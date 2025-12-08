from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class CompanySize(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "enterprise"

class Industry(str, Enum):
    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    RETAIL = "retail"
    MANUFACTURING = "manufacturing"
    EDUCATION = "education"
    CONSULTING = "consulting"
    OTHER = "other"

class JobTitle(str, Enum):
    EXECUTIVE = "executive"
    MANAGER = "manager"
    DIRECTOR = "director"
    VP = "vp"
    INDIVIDUAL_CONTRIBUTOR = "individual_contributor"
    ANALYST = "analyst"
    SPECIALIST = "specialist"

class LeadSource(str, Enum):
    WEBSITE = "website"
    REFERRAL = "referral"
    SOCIAL_MEDIA = "social_media"
    EMAIL_CAMPAIGN = "email_campaign"
    EVENT = "event"
    WEBINAR = "webinar"
    PAID_ADS = "paid_ads"
    ORGANIC_SEARCH = "organic_search"
    DIRECT = "direct"

class BudgetRange(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    ENTERPRISE = "enterprise"

class AuthorityLevel(str, Enum):
    DECISION_MAKER = "decision_maker"
    INFLUENCER = "influencer"
    RECOMMENDER = "recommender"
    USER = "user"

class Timeline(str, Enum):
    IMMEDIATE = "immediate"
    ONE_MONTH = "1_month"
    THREE_MONTHS = "3_months"
    SIX_MONTHS = "6_months"
    ONE_YEAR = "1_year"
    UNKNOWN = "unknown"

class LeadData(BaseModel):
    lead_id: str
    company_size: CompanySize
    industry: Industry
    job_title: JobTitle
    lead_source: LeadSource
    engagement_score: int = Field(ge=0, le=100)
    website_visits: int = Field(ge=0)
    email_opens: int = Field(ge=0)
    form_submissions: int = Field(ge=0)
    demo_requests: int = Field(ge=0)
    content_downloads: int = Field(ge=0)
    social_media_engagement: int = Field(ge=0)
    days_since_last_activity: int = Field(ge=0)
    budget_range: BudgetRange
    authority_level: AuthorityLevel
    timeline: Timeline
    pain_points: List[str] = Field(default_factory=list)
    competitor_usage: bool = False
    marketing_qualified: bool = False
    sales_qualified: bool = False
    created_date: Optional[datetime] = None
    last_activity_date: Optional[datetime] = None

class LeadScore(BaseModel):
    lead_id: str
    score: int = Field(ge=0, le=100)
    conversion_probability: float = Field(ge=0.0, le=1.0)
    model_used: str
    confidence: float = Field(ge=0.0, le=1.0)
    recommendation: str
    factors: Dict[str, Any]
    timestamp: datetime

class BatchLeadScore(BaseModel):
    scores: List[LeadScore]
    total_processed: int
    high_priority: int
    medium_priority: int
    low_priority: int
    processing_time_ms: float

class TrainingResponse(BaseModel):
    message: str
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    training_samples: int
    training_date: datetime

class ModelInfo(BaseModel):
    model_name: str
    version: str
    description: str
    features: List[str]
    algorithms: List[str]
    training_date: Optional[datetime]
    last_updated: datetime
    performance_metrics: Dict[str, float]
    model_status: str

class ModelPerformance(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    confusion_matrix: List[List[int]]
    feature_importance: Dict[str, float]
    training_samples: int
    test_samples: int
    training_date: datetime

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    model_loaded: bool
    version: str

class SampleDataResponse(BaseModel):
    sample_leads: List[LeadData]
    total_generated: int
    summary_stats: Dict[str, Any]
    timestamp: datetime