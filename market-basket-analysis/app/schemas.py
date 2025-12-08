from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


class AlgorithmType(str, Enum):
    APRIORI = "apriori"
    FP_GROWTH = "fp_growth"
    ECLAT = "eclat"


class TransactionItem(BaseModel):
    """Individual item in a transaction"""
    item_id: str = Field(..., description="Unique identifier for the item")
    item_name: str = Field(..., description="Human-readable name of the item")
    category: Optional[str] = Field(None, description="Category of the item")
    price: Optional[float] = Field(None, description="Price of the item")
    quantity: Optional[int] = Field(1, ge=1, description="Quantity of the item")


class Transaction(BaseModel):
    """A complete transaction/basket"""
    transaction_id: str = Field(..., description="Unique identifier for the transaction")
    items: List[TransactionItem] = Field(..., description="List of items in the transaction")
    timestamp: Optional[datetime] = Field(None, description="Transaction timestamp")
    total_amount: Optional[float] = Field(None, description="Total transaction amount")
    customer_id: Optional[str] = Field(None, description="Customer identifier")
    store_id: Optional[str] = Field(None, description="Store identifier")


class AssociationRule(BaseModel):
    """Association rule with metrics"""
    antecedent: List[str] = Field(..., description="Items in the antecedent (if part)")
    consequent: List[str] = Field(..., description="Items in the consequent (then part)")
    support: float = Field(..., description="Support of the rule")
    confidence: float = Field(..., description="Confidence of the rule")
    lift: float = Field(..., description="Lift of the rule")
    conviction: Optional[float] = Field(None, description="Conviction of the rule")
    leverage: Optional[float] = Field(None, description="Leverage of the rule")
    count: int = Field(..., description="Number of transactions supporting this rule")


class FrequentItemset(BaseModel):
    """Frequent itemset with support"""
    items: List[str] = Field(..., description="List of items in the itemset")
    support: float = Field(..., description="Support of the itemset")
    count: int = Field(..., description="Number of transactions containing this itemset")
    length: int = Field(..., description="Number of items in the itemset")


class AnalysisRequest(BaseModel):
    """Request for market basket analysis"""
    transactions: List[Transaction] = Field(..., description="List of transactions to analyze")
    min_support: float = Field(0.01, ge=0.0, le=1.0, description="Minimum support threshold")
    min_confidence: float = Field(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")
    min_lift: Optional[float] = Field(None, ge=0.0, description="Minimum lift threshold")
    max_length: Optional[int] = Field(None, ge=1, description="Maximum length of itemsets")
    algorithm: AlgorithmType = Field(AlgorithmType.APRIORI, description="Algorithm to use")


class AnalysisResponse(BaseModel):
    """Response from market basket analysis"""
    frequent_itemsets: List[FrequentItemset] = Field(..., description="Discovered frequent itemsets")
    association_rules: List[AssociationRule] = Field(..., description="Discovered association rules")
    total_transactions: int = Field(..., description="Total number of transactions analyzed")
    total_items: int = Field(..., description="Total number of unique items")
    algorithm_used: AlgorithmType = Field(..., description="Algorithm used for analysis")
    processing_time_seconds: float = Field(..., description="Time taken for analysis")
    summary_statistics: Dict[str, Any] = Field(..., description="Summary statistics")


class RecommendationRequest(BaseModel):
    """Request for basket recommendations"""
    current_items: List[str] = Field(..., description="Items currently in the basket")
    max_recommendations: int = Field(5, ge=1, le=20, description="Maximum number of recommendations")
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum confidence for recommendations")
    customer_id: Optional[str] = Field(None, description="Customer identifier for personalization")


class BasketRecommendation(BaseModel):
    """Recommended item for basket"""
    item_id: str = Field(..., description="Recommended item ID")
    item_name: str = Field(..., description="Recommended item name")
    confidence: float = Field(..., description="Confidence of the recommendation")
    lift: float = Field(..., description="Lift of the recommendation")
    reason: str = Field(..., description="Explanation for the recommendation")
    estimated_price: Optional[float] = Field(None, description="Estimated price of the item")


class RecommendationResponse(BaseModel):
    """Response for basket recommendations"""
    recommendations: List[BasketRecommendation] = Field(..., description="List of recommended items")
    current_basket_value: Optional[float] = Field(None, description="Current basket value")
    recommended_additional_value: Optional[float] = Field(None, description="Value of recommended items")
    total_basket_value: Optional[float] = Field(None, description="Total estimated basket value")


class PatternDiscoveryRequest(BaseModel):
    """Request for pattern discovery"""
    transactions: List[Transaction] = Field(..., description="Transaction data")
    pattern_type: str = Field(..., description="Type of pattern to discover (sequence, association, etc.)")
    min_support: float = Field(0.01, ge=0.0, le=1.0, description="Minimum support threshold")
    time_window: Optional[int] = Field(None, description="Time window for sequential patterns (in days)")
    customer_segment: Optional[str] = Field(None, description="Customer segment to focus on")


class Pattern(BaseModel):
    """Discovered pattern"""
    pattern_id: str = Field(..., description="Unique identifier for the pattern")
    pattern_type: str = Field(..., description="Type of pattern")
    description: str = Field(..., description="Human-readable description")
    items: List[str] = Field(..., description="Items involved in the pattern")
    support: float = Field(..., description="Support of the pattern")
    confidence: Optional[float] = Field(None, description="Confidence of the pattern")
    lift: Optional[float] = Field(None, description="Lift of the pattern")
    frequency: int = Field(..., description="Frequency of the pattern")
    customer_segments: Optional[List[str]] = Field(None, description="Customer segments where pattern is prominent")


class PatternDiscoveryResponse(BaseModel):
    """Response for pattern discovery"""
    patterns: List[Pattern] = Field(..., description="Discovered patterns")
    total_patterns: int = Field(..., description="Total number of patterns discovered")
    processing_time_seconds: float = Field(..., description="Time taken for pattern discovery")
    summary: Dict[str, Any] = Field(..., description="Summary of findings")


class VisualizationRequest(BaseModel):
    """Request for generating visualizations"""
    analysis_results: AnalysisResponse = Field(..., description="Analysis results to visualize")
    visualization_type: str = Field(..., description="Type of visualization (network, heatmap, etc.)")
    top_n: int = Field(10, ge=1, le=50, description="Number of top rules/itemsets to visualize")


class VisualizationResponse(BaseModel):
    """Response for visualization data"""
    visualization_data: Dict[str, Any] = Field(..., description="Data for visualization")
    chart_type: str = Field(..., description="Type of chart generated")
    description: str = Field(..., description="Description of the visualization")
    recommendations: List[str] = Field(..., description="Recommendations based on the analysis")