from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class OptimizationObjective(str, Enum):
    """Optimization objectives"""
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_PROFIT = "maximize_profit"
    MINIMIZE_TIME = "minimize_time"
    MINIMIZE_RISK = "minimize_risk"
    BALANCE_COST_SERVICE = "balance_cost_service"

class ConstraintType(str, Enum):
    """Types of constraints"""
    CAPACITY = "capacity"
    DEMAND = "demand"
    SUPPLY = "supply"
    TIME = "time"
    BUDGET = "budget"
    QUALITY = "quality"

class NodeType(str, Enum):
    """Supply chain node types"""
    SUPPLIER = "supplier"
    MANUFACTURER = "manufacturer"
    DISTRIBUTOR = "distributor"
    RETAILER = "retailer"
    WAREHOUSE = "warehouse"
    TRANSPORTATION = "transportation"

class TransportMode(str, Enum):
    """Transportation modes"""
    TRUCK = "truck"
    TRAIN = "train"
    SHIP = "ship"
    AIR = "air"
    PIPELINE = "pipeline"

class SupplyChainNode(BaseModel):
    """Supply chain node (facility/location)"""
    node_id: str = Field(..., description="Unique node identifier")
    name: str = Field(..., description="Node name")
    node_type: NodeType = Field(..., description="Type of node")
    location: Dict[str, float] = Field(..., description="Geographic coordinates")
    capacity: float = Field(..., description="Maximum capacity")
    current_inventory: float = Field(default=0.0, description="Current inventory level")
    operating_cost: float = Field(default=0.0, description="Operating cost per unit")
    fixed_cost: float = Field(default=0.0, description="Fixed operating cost")
    setup_cost: float = Field(default=0.0, description="Setup/establishment cost")
    lead_time: int = Field(default=1, description="Lead time in days")
    reliability: float = Field(default=1.0, ge=0.0, le=1.0, description="Reliability factor")
    quality_rating: float = Field(default=1.0, ge=0.0, le=1.0, description="Quality rating")

class SupplyChainEdge(BaseModel):
    """Supply chain edge (transportation link)"""
    edge_id: str = Field(..., description="Unique edge identifier")
    from_node_id: str = Field(..., description="Source node ID")
    to_node_id: str = Field(..., description="Destination node ID")
    transport_mode: TransportMode = Field(..., description="Transportation mode")
    capacity: float = Field(..., description="Maximum transport capacity")
    unit_cost: float = Field(..., description="Cost per unit transported")
    fixed_cost: float = Field(default=0.0, description="Fixed transportation cost")
    transit_time: int = Field(..., description="Transit time in days")
    distance: float = Field(..., description="Distance in kilometers")
    reliability: float = Field(default=1.0, ge=0.0, le=1.0, description="Transportation reliability")
    carbon_emission: float = Field(default=0.0, description="Carbon emission per unit")

class Product(BaseModel):
    """Product information"""
    product_id: str = Field(..., description="Unique product identifier")
    name: str = Field(..., description="Product name")
    unit_cost: float = Field(..., description="Unit production cost")
    unit_price: float = Field(..., description="Unit selling price")
    weight: float = Field(default=1.0, description="Unit weight in kg")
    volume: float = Field(default=1.0, description="Unit volume in cubic meters")
    shelf_life: int = Field(default=365, description="Shelf life in days")
    storage_requirements: str = Field(default="standard", description="Storage requirements")

class DemandForecast(BaseModel):
    """Demand forecast data"""
    product_id: str = Field(..., description="Product identifier")
    node_id: str = Field(..., description="Node where demand occurs")
    period: str = Field(..., description="Time period (YYYY-MM)")
    forecasted_demand: float = Field(..., description="Forecasted demand quantity")
    actual_demand: Optional[float] = Field(None, description="Actual demand quantity")
    confidence_level: float = Field(default=0.95, ge=0.0, le=1.0, description="Forecast confidence")

class InventoryPolicy(BaseModel):
    """Inventory management policy"""
    node_id: str = Field(..., description="Node identifier")
    product_id: str = Field(..., description="Product identifier")
    min_inventory: float = Field(..., description="Minimum inventory level")
    max_inventory: float = Field(..., description="Maximum inventory level")
    reorder_point: float = Field(..., description="Reorder point")
    order_quantity: float = Field(..., description="Order quantity")
    safety_stock: float = Field(default=0.0, description="Safety stock level")
    review_period: int = Field(default=7, description="Review period in days")

class OptimizationRequest(BaseModel):
    """Request schema for supply chain optimization"""
    nodes: List[SupplyChainNode] = Field(..., description="Supply chain nodes")
    edges: List[SupplyChainEdge] = Field(..., description="Supply chain edges")
    products: List[Product] = Field(..., description="Products to optimize")
    demand_forecasts: List[DemandForecast] = Field(..., description="Demand forecasts")
    inventory_policies: Optional[List[InventoryPolicy]] = Field(None, description="Inventory policies")
    objective: OptimizationObjective = Field(..., description="Optimization objective")
    time_horizon: int = Field(default=30, description="Planning horizon in days")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Additional constraints")

class FlowDecision(BaseModel):
    """Flow decision (quantity to move between nodes)"""
    edge_id: str = Field(..., description="Edge identifier")
    product_id: str = Field(..., description="Product identifier")
    quantity: float = Field(..., description="Quantity to transport")
    cost: float = Field(..., description="Total transportation cost")
    time: int = Field(..., description="Transportation time")
    carbon_emission: float = Field(default=0.0, description="Carbon emissions")

class InventoryDecision(BaseModel):
    """Inventory decision (storage quantity at nodes)"""
    node_id: str = Field(..., description="Node identifier")
    product_id: str = Field(..., description="Product identifier")
    quantity: float = Field(..., description="Inventory quantity")
    holding_cost: float = Field(..., description="Holding cost")
    storage_capacity_utilization: float = Field(..., description="Capacity utilization")

class NodeDecision(BaseModel):
    """Node operational decision"""
    node_id: str = Field(..., description="Node identifier")
    is_active: bool = Field(..., description="Whether node should be active")
    production_quantity: float = Field(..., description="Production quantity")
    operating_cost: float = Field(..., description="Operating cost")
    setup_cost: float = Field(default=0.0, description="Setup cost if activated")

class OptimizationResult(BaseModel):
    """Response schema for optimization results"""
    optimization_id: str = Field(..., description="Unique optimization identifier")
    objective_value: float = Field(..., description="Objective function value")
    total_cost: float = Field(..., description="Total supply chain cost")
    total_revenue: float = Field(..., description="Total revenue")
    total_profit: float = Field(..., description="Total profit")
    service_level: float = Field(..., description="Overall service level")
    carbon_footprint: float = Field(..., description="Total carbon footprint")
    flow_decisions: List[FlowDecision] = Field(..., description="Transportation decisions")
    inventory_decisions: List[InventoryDecision] = Field(..., description="Inventory decisions")
    node_decisions: List[NodeDecision] = Field(..., description="Node operational decisions")
    execution_time: float = Field(..., description="Optimization execution time")
    solution_status: str = Field(..., description="Solution status (optimal/feasible/infeasible)")
    recommendations: List[str] = Field(default_factory=list, description="Optimization recommendations")

class NetworkDesignRequest(BaseModel):
    """Request for supply chain network design"""
    candidate_nodes: List[SupplyChainNode] = Field(..., description="Candidate facility locations")
    existing_nodes: List[SupplyChainNode] = Field(default_factory=list, description="Existing facilities")
    edges: List[SupplyChainEdge] = Field(..., description="Potential transportation links")
    products: List[Product] = Field(..., description="Products")
    demand_forecasts: List[DemandForecast] = Field(..., description="Demand forecasts")
    budget_constraint: Optional[float] = Field(None, description="Total budget constraint")
    service_level_target: float = Field(default=0.95, ge=0.0, le=1.0, description="Service level target")
    carbon_target: Optional[float] = Field(None, description="Carbon emission target")

class NetworkDesignResult(BaseModel):
    """Response for network design optimization"""
    design_id: str = Field(..., description="Design identifier")
    selected_nodes: List[SupplyChainNode] = Field(..., description="Selected facilities")
    selected_edges: List[SupplyChainEdge] = Field(..., description="Selected transportation links")
    total_cost: float = Field(..., description="Total network cost")
    fixed_cost: float = Field(..., description="Fixed facility costs")
    operating_cost: float = Field(..., description="Operating costs")
    transportation_cost: float = Field(..., description="Transportation costs")
    service_level: float = Field(..., description="Achieved service level")
    carbon_footprint: float = Field(..., description="Carbon footprint")
    roi: float = Field(..., description="Return on investment")
    payback_period: float = Field(..., description="Payback period in years")
    npv: float = Field(..., description="Net present value")

class RiskAssessment(BaseModel):
    """Supply chain risk assessment"""
    risk_id: str = Field(..., description="Risk identifier")
    risk_type: str = Field(..., description="Type of risk")
    probability: float = Field(..., ge=0.0, le=1.0, description="Risk probability")
    impact: float = Field(..., ge=0.0, le=1.0, description="Risk impact")
    affected_nodes: List[str] = Field(..., description="Affected nodes")
    affected_edges: List[str] = Field(default_factory=list, description="Affected edges")
    mitigation_cost: float = Field(default=0.0, description="Mitigation cost")
    description: str = Field(..., description="Risk description")

class RiskMitigationRequest(BaseModel):
    """Request for risk mitigation optimization"""
    optimization_request: OptimizationRequest = Field(..., description="Base optimization request")
    risks: List[RiskAssessment] = Field(..., description="Identified risks")
    risk_tolerance: float = Field(default=0.1, ge=0.0, le=1.0, description="Risk tolerance level")
    mitigation_budget: Optional[float] = Field(None, description="Risk mitigation budget")

class SustainabilityMetrics(BaseModel):
    """Sustainability and environmental metrics"""
    carbon_emissions: float = Field(..., description="Total carbon emissions (kg CO2e)")
    energy_consumption: float = Field(..., description="Energy consumption (kWh)")
    water_usage: float = Field(..., description="Water usage (liters)")
    waste_generated: float = Field(..., description="Waste generated (kg)")
    recycling_rate: float = Field(..., ge=0.0, le=1.0, description="Recycling rate")
    renewable_energy_ratio: float = Field(..., ge=0.0, le=1.0, description="Renewable energy ratio")

class SustainabilityOptimizationRequest(BaseModel):
    """Request for sustainability-focused optimization"""
    optimization_request: OptimizationRequest = Field(..., description="Base optimization request")
    sustainability_targets: SustainabilityMetrics = Field(..., description="Sustainability targets")
    carbon_price: float = Field(default=0.0, description="Carbon price per kg CO2e")
    sustainability_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Weight for sustainability in objective")