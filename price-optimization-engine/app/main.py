from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import numpy as np
from datetime import datetime

from .model import PriceOptimizationModel
from .schemas import (
    PriceRequest, PriceResponse, BatchPriceRequest, BatchPriceResponse,
    MarketData, PricingStrategy, PriceElasticity, CompetitiveAnalysis,
    ModelPerformance, OptimizationConstraints
)

app = FastAPI(
    title="Price Optimization Engine API",
    description="AI-powered price optimization using Reinforcement Learning and Machine Learning",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

def get_model():
    """Get or initialize the price optimization model"""
    global model
    if model is None:
        model = PriceOptimizationModel()
        # Train model with synthetic data on initialization
        synthetic_data = model.generate_synthetic_market_data(500)
        model.train_models(synthetic_data)
    return model

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Price Optimization Engine API",
        "version": "1.0.0",
        "status": "active",
        "available_strategies": [strategy.value for strategy in PricingStrategy],
        "features": [
            "Reinforcement Learning Optimization",
            "Machine Learning Demand Prediction",
            "Competitive Pricing Analysis",
            "Price Elasticity Calculation",
            "Batch Price Optimization"
        ]
    }

@app.post("/optimize-price", response_model=PriceResponse)
async def optimize_price(request: PriceRequest):
    """
    Optimize price for a single product using AI/ML
    
    Args:
        request: Price optimization request
        
    Returns:
        Optimized price recommendation
    """
    try:
        model = get_model()
        
        # Use reinforcement learning as primary strategy
        response = model.optimize_price_with_rl(request)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Price optimization error: {str(e)}')

@app.post("/optimize-price/{strategy}", response_model=PriceResponse)
async def optimize_price_with_strategy(strategy: str, request: PriceRequest):
    """
    Optimize price using specific strategy
    
    Args:
        strategy: Pricing strategy to use
        request: Price optimization request
        
    Returns:
        Optimized price recommendation
    """
    try:
        model = get_model()
        
        # Map strategy string to enum
        strategy_enum = PricingStrategy(strategy)
        
        if strategy_enum == PricingStrategy.REINFORCEMENT_LEARNING:
            response = model.optimize_price_with_rl(request)
        elif strategy_enum == PricingStrategy.DYNAMIC_PRICING:
            response = model.optimize_price_with_ml(request)
        else:
            # Fallback to rule-based for other strategies
            response = model._rule_based_optimization(request)
        
        return response
        
    except ValueError:
        raise HTTPException(status_code=400, detail=f'Invalid strategy: {strategy}')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Price optimization error: {str(e)}')

@app.post("/optimize-batch", response_model=BatchPriceResponse)
async def optimize_batch_prices(request: BatchPriceRequest):
    """
    Optimize prices for multiple products
    
    Args:
        request: Batch price optimization request
        
    Returns:
        Batch price optimization results
    """
    try:
        model = get_model()
        
        start_time = datetime.now()
        
        # Optimize each product
        optimized_prices = model.optimize_batch_prices(
            request.products,
            request.optimization_strategy,
            request.max_price_change
        )
        
        # Calculate total expected revenue
        total_revenue = sum(price.expected_revenue for price in optimized_prices)
        
        # Check for constraint violations
        violations = []
        for i, (product, optimized_price) in enumerate(zip(request.products, optimized_prices)):
            price_change = abs(optimized_price.optimized_price - product.current_price) / product.current_price
            if price_change > request.max_price_change:
                violations.append(f"Product {product.product_id}: Price change {price_change:.1%} exceeds limit")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return BatchPriceResponse(
            products=optimized_prices,
            total_expected_revenue=total_revenue,
            optimization_time=processing_time,
            strategy_used=request.optimization_strategy,
            constraint_violations=violations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Batch optimization error: {str(e)}')

@app.post("/analyze-elasticity", response_model=PriceElasticity)
async def analyze_price_elasticity(
    product_id: str,
    price_history: List[float],
    demand_history: List[float]
):
    """
    Analyze price elasticity for a product
    
    Args:
        product_id: Product identifier
        price_history: Historical prices
        demand_history: Historical demand data
        
    Returns:
        Price elasticity analysis
    """
    try:
        model = get_model()
        
        elasticity = model.calculate_price_elasticity(price_history, demand_history)
        elasticity.product_id = product_id
        
        return elasticity
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Elasticity analysis error: {str(e)}')

@app.post("/analyze-competition", response_model=CompetitiveAnalysis)
async def analyze_competitive_position(
    product_id: str,
    product_price: float,
    competitor_prices: List[float]
):
    """
    Analyze competitive pricing position
    
    Args:
        product_id: Product identifier
        product_price: Current product price
        competitor_prices: List of competitor prices
        
    Returns:
        Competitive analysis results
    """
    try:
        model = get_model()
        
        analysis = model.analyze_competitive_position(product_price, competitor_prices)
        analysis.product_id = product_id
        
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Competitive analysis error: {str(e)}')

@app.post("/train-model")
async def train_model(n_samples: int = 1000):
    """
    Train the price optimization models
    
    Args:
        n_samples: Number of synthetic samples to generate for training
        
    Returns:
        Training results
    """
    try:
        model = get_model()
        
        # Generate synthetic training data
        synthetic_data = model.generate_synthetic_market_data(n_samples)
        
        # Train models
        model.train_models(synthetic_data)
        
        return {
            "message": "Models trained successfully",
            "training_samples": n_samples,
            "models_trained": ["demand", "elasticity", "competitive"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Training error: {str(e)}')

@app.get("/generate-sample-data")
async def generate_sample_data(n_samples: int = 100):
    """
    Generate sample market data for testing
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Sample market data
    """
    try:
        model = get_model()
        
        sample_data = model.generate_synthetic_market_data(n_samples)
        
        return {
            "sample_data": [data.dict() for data in sample_data],
            "count": len(sample_data),
            "categories": list(set(data.product_id.split("_")[0] for data in sample_data))
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Sample data generation error: {str(e)}')

@app.get("/model-performance")
async def get_model_performance():
    """
    Get model performance metrics
    
    Returns:
        Model performance information
    """
    try:
        model = get_model()
        
        # Return basic model info
        return {
            "models_status": {
                "demand_model": "trained" if model.is_trained else "not_trained",
                "elasticity_model": "trained" if model.is_trained else "not_trained",
                "rl_agent": "active",
                "q_table_size": len(model.rl_agent.q_table)
            },
            "training_history": {
                "rl_experiences": len(model.rl_agent.state_history),
                "is_trained": model.is_trained
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model performance error: {str(e)}")

@app.get("/pricing-strategies")
async def get_pricing_strategies():
    """
    Get available pricing strategies
    
    Returns:
        List of available pricing strategies
    """
    return {
        "strategies": [
            {
                "name": strategy.value,
                "description": _get_strategy_description(strategy)
            }
            for strategy in PricingStrategy
        ]
    }

def _get_strategy_description(strategy: PricingStrategy) -> str:
    """Get description for pricing strategy"""
    descriptions = {
        PricingStrategy.REINFORCEMENT_LEARNING: "AI-powered optimization using Q-learning",
        PricingStrategy.DYNAMIC_PRICING: "ML-based demand prediction and optimization",
        PricingStrategy.COMPETITIVE_PRICING: "Competitor-based pricing strategy",
        PricingStrategy.DEMAND_BASED: "Demand-driven pricing adjustments",
        PricingStrategy.COST_PLUS: "Cost-plus margin pricing"
    }
    return descriptions.get(strategy, "Unknown strategy")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        model = get_model()
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "rl_agent_active": model.rl_agent is not None,
            "available_strategies": [strategy.value for strategy in PricingStrategy],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009)