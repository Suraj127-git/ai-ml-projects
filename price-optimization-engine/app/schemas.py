from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union
from datetime import datetime
from enum import Enum

class ProductCategory(str, Enum):
    """Product categories for pricing"""
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    FOOD = "food"
    BOOKS = "books"
    HOME_KITCHEN = "home_kitchen"
    SPORTS = "sports"
    BEAUTY = "beauty"
    TOYS = "toys"
    AUTOMOTIVE = "automotive"
    HEALTH = "health"

class PricingStrategy(str, Enum):
    """Pricing strategies"""
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    DYNAMIC_PRICING = "dynamic_pricing"
    COMPETITIVE_PRICING = "competitive_pricing"
    DEMAND_BASED = "demand_based"
    COST_PLUS = "cost_plus"

class PriceRequest(BaseModel):
    """Request schema for price optimization"""
    product_id: str = Field(..., description="Unique product identifier")
    product_name: str = Field(..., description="Product name")
    category: ProductCategory = Field(..., description="Product category")
    current_price: float = Field(..., gt=0, description="Current product price")
    cost_price: float = Field(..., gt=0, description="Product cost price")
    competitor_prices: List[float] = Field(default=[], description="Competitor prices")
    demand_history: List[float] = Field(default=[], description="Historical demand data")
    inventory_level: int = Field(default=0, ge=0, description="Current inventory level")
    seasonality_factor: float = Field(default=1.0, ge=0.1, le=3.0, description="Seasonality multiplier")
    price_elasticity: float = Field(default=-1.0, ge=-5.0, le=0.0, description="Price elasticity coefficient")
    target_margin: float = Field(default=0.2, ge=0.0, le=0.8, description="Target profit margin")
    market_conditions: str = Field(default="normal", description="Market conditions")

class PriceResponse(BaseModel):
    """Response schema for price optimization"""
    product_id: str = Field(..., description="Product identifier")
    optimized_price: float = Field(..., description="Recommended optimal price")
    current_price: float = Field(..., description="Current price for comparison")
    expected_revenue: float = Field(..., description="Expected revenue at optimized price")
    expected_demand: float = Field(..., description="Expected demand at optimized price")
    profit_margin: float = Field(..., description="Profit margin at optimized price")
    confidence_score: float = Field(..., description="Confidence in the recommendation")
    reasoning: str = Field(..., description="Explanation for the price recommendation")
    strategy_used: PricingStrategy = Field(..., description="Pricing strategy employed")
    timestamp: str = Field(..., description="Recommendation timestamp")

class BatchPriceRequest(BaseModel):
    """Request schema for batch price optimization"""
    products: List[PriceRequest] = Field(..., description="List of products to optimize")
    optimization_strategy: PricingStrategy = Field(default=PricingStrategy.REINFORCEMENT_LEARNING, description="Overall optimization strategy")
    max_price_change: float = Field(default=0.2, ge=0.0, le=1.0, description="Maximum allowed price change percentage")
    business_constraints: Dict[str, Union[float, int]] = Field(default={}, description="Business constraints")

class BatchPriceResponse(BaseModel):
    """Response schema for batch price optimization"""
    products: List[PriceResponse] = Field(..., description="Optimized prices for all products")
    total_expected_revenue: float = Field(..., description="Total expected revenue")
    optimization_time: float = Field(..., description="Time taken for optimization")
    strategy_used: PricingStrategy = Field(..., description="Strategy used for optimization")
    constraint_violations: List[str] = Field(default=[], description="Any constraint violations")

class MarketData(BaseModel):
    """Market data for pricing analysis"""
    product_id: str = Field(..., description="Product identifier")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    price: float = Field(..., gt=0, description="Product price")
    demand: int = Field(..., ge=0, description="Units sold")
    competitor_price: float = Field(default=0, ge=0, description="Average competitor price")
    inventory: int = Field(default=0, ge=0, description="Inventory level")
    market_share: float = Field(default=0.0, ge=0.0, le=1.0, description="Market share")
    customer_satisfaction: float = Field(default=0.0, ge=0.0, le=1.0, description="Customer satisfaction score")

class ReinforcementLearningState(BaseModel):
    """State representation for reinforcement learning"""
    product_id: str = Field(..., description="Product identifier")
    current_price: float = Field(..., description="Current price")
    cost_price: float = Field(..., description="Cost price")
    competitor_avg_price: float = Field(..., description="Average competitor price")
    inventory_level: int = Field(..., description="Current inventory")
    demand_trend: float = Field(..., description="Demand trend (positive/negative)")
    seasonality_index: float = Field(..., description="Seasonality index")
    price_elasticity: float = Field(..., description="Price elasticity")
    days_since_price_change: int = Field(..., description="Days since last price change")
    market_volatility: float = Field(..., description="Market volatility measure")

class ReinforcementLearningAction(BaseModel):
    """Action representation for reinforcement learning"""
    price_change: float = Field(..., description="Price change amount")
    price_change_percentage: float = Field(..., description="Price change percentage")
    action_type: str = Field(..., description="Type of action (increase/decrease/maintain)")

class ReinforcementLearningReward(BaseModel):
    """Reward calculation for reinforcement learning"""
    revenue_reward: float = Field(..., description="Reward from revenue")
    profit_reward: float = Field(..., description="Reward from profit")
    market_share_reward: float = Field(..., description="Reward from market share")
    customer_satisfaction_reward: float = Field(..., description="Reward from customer satisfaction")
    total_reward: float = Field(..., description="Total reward")

class TrainingData(BaseModel):
    """Training data for price optimization models"""
    product_id: str = Field(..., description="Product identifier")
    features: List[float] = Field(..., description="Feature vector")
    optimal_price: float = Field(..., description="Optimal price")
    reward: float = Field(..., description="Reward achieved")
    action_taken: str = Field(..., description="Action that was taken")

class ModelPerformance(BaseModel):
    """Model performance metrics"""
    model_type: str = Field(..., description="Type of model")
    accuracy: float = Field(..., description="Model accuracy")
    precision: float = Field(..., description="Model precision")
    recall: float = Field(..., description="Model recall")
    f1_score: float = Field(..., description="F1 score")
    training_samples: int = Field(..., description="Number of training samples")
    validation_samples: int = Field(..., description="Number of validation samples")
    training_time: float = Field(..., description="Training time in seconds")
    last_updated: str = Field(..., description="Last model update")

class PriceElasticity(BaseModel):
    """Price elasticity analysis"""
    product_id: str = Field(..., description="Product identifier")
    elasticity_coefficient: float = Field(..., description="Price elasticity coefficient")
    elasticity_category: str = Field(..., description="Elasticity category (elastic/inelastic)")
    confidence_interval: List[float] = Field(..., description="Confidence interval for elasticity")
    statistical_significance: float = Field(..., description="Statistical significance")

class CompetitiveAnalysis(BaseModel):
    """Competitive pricing analysis"""
    product_id: str = Field(..., description="Product identifier")
    competitor_prices: List[float] = Field(..., description="Competitor prices")
    price_position: str = Field(..., description="Price position (premium/parity/discount)")
    competitive_advantage: float = Field(..., description="Competitive advantage score")
    market_pricing_trend: str = Field(..., description="Market pricing trend")

class OptimizationConstraints(BaseModel):
    """Business constraints for price optimization"""
    min_margin: float = Field(default=0.1, ge=0.0, description="Minimum profit margin")
    max_price_change: float = Field(default=0.2, ge=0.0, description="Maximum price change percentage")
    min_price: float = Field(default=0.0, ge=0.0, description="Minimum absolute price")
    max_price: float = Field(default=999999.0, description="Maximum absolute price")
    competitor_price_buffer: float = Field(default=0.05, ge=0.0, description="Competitor price buffer")
    inventory_target: int = Field(default=100, ge=0, description="Target inventory level")

class ABRTestResult(BaseModel):
    """A/B test result for pricing strategies"""
    test_id: str = Field(..., description="Test identifier")
    product_id: str = Field(..., description="Product identifier")
    variant_a: Dict = Field(..., description="Variant A (control) data")
    variant_b: Dict = Field(..., description="Variant B (treatment) data")
    winner: str = Field(..., description="Winning variant")
    statistical_significance: float = Field(..., description="Statistical significance")
    confidence_level: float = Field(..., description="Confidence level")
    recommendation: str = Field(..., description="Test recommendation")