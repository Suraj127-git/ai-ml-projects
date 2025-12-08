from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import date, datetime
from enum import Enum

class OptimizationMethod(str, Enum):
    EOQ = "eoq"  # Economic Order Quantity
    ROP = "rop"  # Reorder Point
    SAFETY_STOCK = "safety_stock"
    ABC_ANALYSIS = "abc_analysis"
    MULTI_ECHELON = "multi_echelon"

class InventoryPolicy(str, Enum):
    CONTINUOUS_REVIEW = "continuous_review"
    PERIODIC_REVIEW = "periodic_review"
    BASE_STOCK = "base_stock"
    S_S_POLICY = "s_s_policy"  # (s, S) policy

class ProductData(BaseModel):
    product_id: str
    product_name: str
    category: str
    unit_cost: float = Field(gt=0)
    holding_cost_rate: float = Field(ge=0, le=1)
    ordering_cost: float = Field(ge=0)
    lead_time_days: int = Field(ge=0)
    current_stock: int = Field(ge=0)
    demand_rate: float = Field(gt=0)  # units per day
    demand_std: Optional[float] = Field(None, ge=0)
    service_level: float = Field(default=0.95, ge=0.8, le=0.99)
    annual_demand: Optional[float] = Field(None, gt=0)
    annual_revenue: Optional[float] = Field(None, gt=0)
    expiration_days: Optional[int] = Field(None, ge=1)
    min_order_quantity: int = Field(default=1, ge=1)
    max_stock_capacity: Optional[int] = Field(None, ge=1)
    supplier_reliability: float = Field(default=0.95, ge=0, le=1)

class HistoricalDemand(BaseModel):
    date: date
    demand: float = Field(ge=0)
    stock_out: bool = Field(default=False)
    lost_sales: Optional[float] = Field(None, ge=0)

class InventoryData(BaseModel):
    product_id: str
    current_data: ProductData
    historical_demand: List[HistoricalDemand]
    
    class Config:
        json_encoders = {
            date: lambda v: v.isoformat()
        }

class OptimizationRequest(BaseModel):
    product_data: ProductData
    optimization_method: OptimizationMethod = OptimizationMethod.EOQ
    service_level: Optional[float] = Field(None, ge=0.8, le=0.99)
    forecast_horizon_days: int = Field(default=30, ge=7, le=365)
    include_demand_forecast: bool = Field(default=True)
    demand_volatility_factor: float = Field(default=1.0, ge=0.5, le=3.0)

class OptimizationResult(BaseModel):
    product_id: str
    optimization_method: str
    economic_order_quantity: int
    reorder_point: int
    safety_stock: int
    average_inventory: float
    total_cost: float
    ordering_cost: float
    holding_cost: float
    stockout_cost: float = Field(default=0.0)
    service_level_achieved: float
    inventory_turnover: float
    days_of_supply: int
    recommendations: List[str]
    confidence_score: float = Field(ge=0, le=1)
    calculation_date: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class BatchOptimizationRequest(BaseModel):
    products: List[OptimizationRequest]
    optimization_method: OptimizationMethod = OptimizationMethod.EOQ
    
class BatchOptimizationResponse(BaseModel):
    results: List[OptimizationResult]
    total_products: int
    processing_time: float
    summary_stats: Dict[str, float]

class ABCAnalysisRequest(BaseModel):
    products: List[ProductData]
    historical_data: List[HistoricalDemand]
    revenue_percentage_a: float = Field(default=0.8, ge=0.6, le=0.95)
    item_percentage_a: float = Field(default=0.2, ge=0.1, le=0.4)
    
class ABCAnalysisResult(BaseModel):
    product_id: str
    category: str  # A, B, or C
    annual_revenue: float
    revenue_percentage: float
    demand_volume: float
    optimization_priority: str
    recommended_policy: str

class ABCAnalysisResponse(BaseModel):
    analysis_results: List[ABCAnalysisResult]
    category_summary: Dict[str, int]
    total_revenue: float
    recommendations: List[str]

class MultiEchelonRequest(BaseModel):
    warehouses: List[str]
    products: List[ProductData]
    transportation_costs: Dict[str, float]
    transfer_lead_times: Dict[str, int]
    central_warehouse_capacity: int
    
class MultiEchelonResult(BaseModel):
    warehouse_id: str
    product_id: str
    optimal_stock_level: int
    transfer_quantity: int
    total_cost: float
    service_level: float
    stock_transfer_recommendations: List[str]

class MultiEchelonResponse(BaseModel):
    echelon_results: List[MultiEchelonResult]
    total_network_cost: float
    overall_service_level: float
    transfer_plan: List[Dict[str, Any]]

class StockAlert(BaseModel):
    product_id: str
    alert_type: str  # "low_stock", "overstock", "expiring", "reorder_needed"
    current_stock: int
    recommended_action: str
    urgency_level: str  # "low", "medium", "high", "critical"
    estimated_impact: float
    action_deadline: Optional[date] = None
    
    class Config:
        json_encoders = {
            date: lambda v: v.isoformat()
        }

class StockAlertResponse(BaseModel):
    alerts: List[StockAlert]
    total_alerts: int
    critical_alerts: int
    summary_by_type: Dict[str, int]
    recommended_actions: List[str]

class DemandForecast(BaseModel):
    date: date
    predicted_demand: float
    confidence_lower: float
    confidence_upper: float
    
class InventoryPolicy(BaseModel):
    product_id: str
    policy_type: InventoryPolicy
    order_quantity: int
    reorder_point: int
    review_period_days: Optional[int] = Field(None, ge=1)
    max_inventory_level: Optional[int] = Field(None, ge=1)
    safety_stock: int
    total_cost: float
    service_level: float

class PolicyOptimizationRequest(BaseModel):
    product_data: ProductData
    available_policies: List[InventoryPolicy]
    optimization_criteria: str = Field(default="total_cost", pattern="^(total_cost|service_level|inventory_turnover)$")
    
class PolicyOptimizationResult(BaseModel):
    recommended_policy: InventoryPolicy
    alternative_policies: List[InventoryPolicy]
    cost_comparison: Dict[str, float]
    service_level_comparison: Dict[str, float]
    recommendation_reason: str

class HealthResponse(BaseModel):
    status: str
    available_methods: List[str]
    total_products_optimized: int
    active_alerts: int
    timestamp: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class OptimizationMetrics(BaseModel):
    total_inventory_value: float
    average_service_level: float
    inventory_turnover_ratio: float
    stockout_rate: float
    holding_cost_percentage: float
    ordering_frequency: float
    optimization_savings: float