from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

from .model import SupplyChainOptimizer
from .schemas import (
    SupplyChainNode, SupplyChainEdge, Product, DemandForecast,
    OptimizationRequest, OptimizationResult, OptimizationObjective,
    NodeType, TransportMode
)

app = FastAPI(
    title="Supply Chain Optimization API",
    description="API for supply chain network optimization, facility location, and inventory management",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize optimizer
optimizer = SupplyChainOptimizer()

@app.post("/optimize-network", response_model=OptimizationResult)
async def optimize_network(request: OptimizationRequest):
    """
    Optimize supply chain network flow
    
    This endpoint solves the network flow optimization problem to minimize costs,
    time, or risk while satisfying all demand constraints.
    """
    try:
        result = optimizer.solve_network_flow_optimization(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Network optimization failed: {str(e)}")

@app.post("/optimize-facility-location")
async def optimize_facility_location(
    nodes: List[SupplyChainNode],
    demands: List[DemandForecast],
    budget_constraint: float
):
    """
    Optimize facility locations using mixed-integer programming
    
    Args:
        nodes: List of potential facility locations with costs and capacities
        demands: List of demand points with quantities
        budget_constraint: Maximum budget for facility setup
    """
    try:
        result = optimizer.optimize_facility_location(nodes, demands, budget_constraint)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Facility location optimization failed: {str(e)}")

@app.post("/optimize-inventory")
async def optimize_inventory(
    demand_forecasts: List[DemandForecast],
    holding_cost_rate: float = 0.2,
    shortage_cost_rate: float = 0.5
):
    """
    Optimize inventory levels using newsvendor model
    
    Args:
        demand_forecasts: List of demand forecasts with mean and std deviation
        holding_cost_rate: Cost of holding inventory (% of value)
        shortage_cost_rate: Cost of shortage (% of value)
    """
    try:
        result = optimizer.optimize_inventory_levels(
            demand_forecasts, holding_cost_rate, shortage_cost_rate
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inventory optimization failed: {str(e)}")

@app.post("/train-demand-forecast")
async def train_demand_forecast(file: Optional[bytes] = None):
    """
    Train demand forecasting models using historical data
    
    Args:
        file: CSV file with historical demand data (optional)
              Expected columns: date, demand, price, seasonality, promotion, competitor_price, economic_index
    """
    try:
        if file:
            # Read uploaded file
            df = pd.read_csv(file)
        else:
            # Generate synthetic data for demonstration
            dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
            df = pd.DataFrame({
                'date': dates,
                'demand': np.random.poisson(100, len(dates)),
                'price': np.random.uniform(10, 100, len(dates)),
                'seasonality': np.sin(2 * np.pi * np.arange(len(dates)) / 365.25),
                'promotion': np.random.choice([0, 1], len(dates), p=[0.8, 0.2]),
                'competitor_price': np.random.uniform(8, 120, len(dates)),
                'economic_index': np.random.uniform(0.9, 1.1, len(dates))
            })
        
        result = optimizer.train_demand_forecasting_model(df)
        
        # Convert models to serializable format
        serializable_result = {}
        for model_name, metrics in result.items():
            serializable_result[model_name] = {
                'mse': float(metrics['mse']),
                'mae': float(metrics['mae']),
                'rmse': float(metrics['rmse'])
            }
        
        return {
            "training_results": serializable_result,
            "training_samples": len(df),
            "training_date": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demand forecast training failed: {str(e)}")

@app.post("/predict-demand")
async def predict_demand(
    product_features: Dict,
    model_type: str = "random_forest"
):
    """
    Predict demand for a product using trained models
    
    Args:
        product_features: Dictionary with product features
            - price: Product price
            - seasonality: Seasonality factor (-1 to 1)
            - promotion: Promotion indicator (0 or 1)
            - competitor_price: Competitor price
            - economic_index: Economic index (0.8 to 1.2)
        model_type: Type of model to use (random_forest, linear_regression)
    """
    try:
        demand = optimizer.predict_demand(product_features, model_type)
        return {
            "predicted_demand": demand,
            "model_type": model_type,
            "prediction_date": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demand prediction failed: {str(e)}")

@app.get("/generate-sample-network")
async def generate_sample_network():
    """
    Generate a sample supply chain network for testing
    
    Returns:
        Sample network with nodes, edges, and demands
    """
    try:
        # Sample nodes
        nodes = [
            SupplyChainNode(
                node_id="supplier_1",
                name="Primary Supplier",
                node_type=NodeType.SUPPLIER,
                location={"lat": 40.7128, "lng": -74.0060},
                capacity=1000,
                setup_cost=50000
            ),
            SupplyChainNode(
                node_id="dc_1",
                name="East Coast Distribution Center",
                node_type=NodeType.DISTRIBUTION_CENTER,
                location={"lat": 39.9526, "lng": -75.1652},
                capacity=800,
                setup_cost=100000
            ),
            SupplyChainNode(
                node_id="dc_2",
                name="West Coast Distribution Center",
                node_type=NodeType.DISTRIBUTION_CENTER,
                location={"lat": 34.0522, "lng": -118.2437},
                capacity=700,
                setup_cost=120000
            ),
            SupplyChainNode(
                node_id="retailer_1",
                name="NYC Retailer",
                node_type=NodeType.RETAILER,
                location={"lat": 40.7128, "lng": -74.0060},
                capacity=300,
                setup_cost=20000
            ),
            SupplyChainNode(
                node_id="retailer_2",
                name="LA Retailer",
                node_type=NodeType.RETAILER,
                location={"lat": 34.0522, "lng": -118.2437},
                capacity=250,
                setup_cost=18000
            )
        ]
        
        # Sample edges
        edges = [
            SupplyChainEdge(
                from_node="supplier_1",
                to_node="dc_1",
                capacity=500,
                transportation_cost=2.5,
                lead_time=3,
                distance=150,
                carbon_factor=0.1,
                reliability=0.95
            ),
            SupplyChainEdge(
                from_node="supplier_1",
                to_node="dc_2",
                capacity=400,
                transportation_cost=3.2,
                lead_time=5,
                distance=280,
                carbon_factor=0.12,
                reliability=0.92
            ),
            SupplyChainEdge(
                from_node="dc_1",
                to_node="retailer_1",
                capacity=200,
                transportation_cost=1.8,
                lead_time=1,
                distance=80,
                carbon_factor=0.08,
                reliability=0.98
            ),
            SupplyChainEdge(
                from_node="dc_2",
                to_node="retailer_2",
                capacity=180,
                transportation_cost=2.1,
                lead_time=2,
                distance=50,
                carbon_factor=0.09,
                reliability=0.96
            )
        ]
        
        # Sample demands
        demands = [
            DemandForecast(
                node_id="retailer_1",
                product_id="product_1",
                quantity=150,
                mean_demand=150,
                std_demand=20,
                probability_exceeding=0.1
            ),
            DemandForecast(
                node_id="retailer_2",
                product_id="product_1",
                quantity=120,
                mean_demand=120,
                std_demand=15,
                probability_exceeding=0.08
            )
        ]
        
        return {
            "nodes": nodes,
            "edges": edges,
            "demands": demands,
            "network_info": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "total_demand": sum(d.quantity for d in demands),
                "generation_date": datetime.now().isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sample network generation failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "supply-chain-optimization",
        "timestamp": datetime.now().isoformat(),
        "pulp_available": optimizer.models.get('pulp_available', False),
        "models_loaded": len(optimizer.models)
    }

@app.get("/model-info")
async def get_model_info():
    """Get information about loaded models"""
    return {
        "models": list(optimizer.models.keys()),
        "scalers": list(optimizer.scalers.keys()),
        "network_data_loaded": optimizer.network_data is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)