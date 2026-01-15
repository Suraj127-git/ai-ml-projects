from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import uvicorn
import os
import pandas as pd
import numpy as np

from model import InventoryOptimizationModel
from schemas import (
    ProductData, InventoryData, OptimizationRequest, OptimizationResult,
    BatchOptimizationRequest, BatchOptimizationResponse, ABCAnalysisRequest,
    ABCAnalysisResponse, MultiEchelonRequest, MultiEchelonResponse,
    StockAlert, StockAlertResponse, HealthResponse, OptimizationMetrics,
    HistoricalDemand
)

# Initialize FastAPI app
app = FastAPI(
    title="Inventory Optimization System API",
    description="API for optimizing inventory levels using EOQ, safety stock, ABC analysis, and multi-echelon optimization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
inventory_model = InventoryOptimizationModel()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Inventory Optimization System API",
        "version": "1.0.0",
        "available_methods": ["EOQ", "Safety Stock", "ABC Analysis", "Multi-Echelon"],
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        available_methods=["EOQ", "Safety Stock", "ABC Analysis", "Multi-Echelon"],
        total_products_optimized=len(inventory_model.abc_categories),
        active_alerts=0,  # Simplified for demo
        timestamp=datetime.now()
    )

@app.post("/optimize", response_model=OptimizationResult)
async def optimize_inventory(request: OptimizationRequest):
    """Optimize inventory for a single product"""
    try:
        # Convert product data to dict
        product_dict = request.product_data.dict()
        
        # Optimize inventory policy
        result = inventory_model.optimize_inventory_policy(
            product_data=product_dict,
            optimization_method=request.optimization_method.value,
            service_level=request.service_level or product_dict['service_level']
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return OptimizationResult(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")

@app.post("/optimize/batch", response_model=BatchOptimizationResponse)
async def batch_optimize_inventory(request: BatchOptimizationRequest):
    """Optimize inventory for multiple products"""
    start_time = datetime.now()
    results = []
    
    try:
        for optimization_request in request.products:
            product_dict = optimization_request.product_data.dict()
            
            result = inventory_model.optimize_inventory_policy(
                product_data=product_dict,
                optimization_method=request.optimization_method.value,
                service_level=optimization_request.service_level or product_dict['service_level']
            )
            
            if 'error' not in result:
                results.append(OptimizationResult(**result))
        
        # Calculate summary statistics
        if results:
            total_cost = sum(r.total_cost for r in results)
            avg_service_level = sum(r.service_level_achieved for r in results) / len(results)
            total_inventory_value = sum(r.average_inventory * 
                optimization_request.product_data.unit_cost for r, optimization_request 
                in zip(results, request.products))
            
            summary_stats = {
                'total_inventory_value': total_inventory_value,
                'average_service_level': avg_service_level,
                'total_annual_cost': total_cost,
                'products_optimized': len(results)
            }
        else:
            summary_stats = {}
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return BatchOptimizationResponse(
            results=results,
            total_products=len(results),
            processing_time=processing_time,
            summary_stats=summary_stats
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch optimization error: {str(e)}")

@app.post("/abc-analysis", response_model=ABCAnalysisResponse)
async def abc_analysis(request: ABCAnalysisRequest):
    """Perform ABC analysis on products"""
    try:
        # Convert products to DataFrame
        products_data = [product.dict() for product in request.products]
        products_df = pd.DataFrame(products_data)
        
        # Perform ABC analysis
        analysis_results = inventory_model.perform_abc_analysis(
            products_df=products_df,
            revenue_percentage_a=request.revenue_percentage_a,
            item_percentage_a=request.item_percentage_a
        )
        
        # Calculate category summary
        category_summary = {}
        for result in analysis_results:
            category = result['category']
            category_summary[category] = category_summary.get(category, 0) + 1
        
        # Generate recommendations
        recommendations = []
        
        if category_summary.get('A', 0) > 0:
            recommendations.append("Focus on tight inventory control for Category A items")
            recommendations.append("Implement continuous review system for high-value items")
        
        if category_summary.get('C', 0) > len(request.products) * 0.5:
            recommendations.append("Consider reducing variety of Category C items")
            recommendations.append("Implement simple control mechanisms for low-value items")
        
        total_revenue = sum(result['annual_revenue'] for result in analysis_results)
        
        return ABCAnalysisResponse(
            analysis_results=[ABCAnalysisResult(**result) for result in analysis_results],
            category_summary=category_summary,
            total_revenue=total_revenue,
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'ABC analysis error: {str(e)}')

@app.post("/multi-echelon", response_model=MultiEchelonResponse)
async def multi_echelon_optimization(request: MultiEchelonRequest):
    """Optimize multi-echelon inventory system"""
    try:
        # Convert products to list of dicts
        products_data = [product.dict() for product in request.products]
        
        # Perform multi-echelon optimization
        echelon_results = inventory_model.optimize_multi_echelon(
            warehouses=request.warehouses,
            products_data=products_data,
            transportation_costs=request.transportation_costs,
            transfer_lead_times=request.transfer_lead_times,
            central_warehouse_capacity=request.central_warehouse_capacity
        )
        
        # Calculate total network cost
        total_network_cost = sum(result['total_cost'] for result in echelon_results)
        
        # Calculate overall service level (simplified)
        overall_service_level = sum(result['service_level'] for result in echelon_results) / len(echelon_results)
        
        # Create transfer plan
        transfer_plan = []
        for result in echelon_results:
            if result['transfer_quantity'] > 0:
                transfer_plan.append({
                    'from_warehouse': 'central',
                    'to_warehouse': result['warehouse_id'],
                    'product_id': result['product_id'],
                    'quantity': result['transfer_quantity'],
                    'cost': result['total_cost']
                })
        
        return MultiEchelonResponse(
            echelon_results=[MultiEchelonResult(**result) for result in echelon_results],
            total_network_cost=total_network_cost,
            overall_service_level=overall_service_level,
            transfer_plan=transfer_plan
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-echelon optimization error: {str(e)}")

@app.post("/stock-alerts", response_model=StockAlertResponse)
async def stock_alerts(request: ABCAnalysisRequest):
    """Generate stock alerts based on current inventory levels"""
    try:
        # Convert products and historical data to DataFrames
        products_data = [product.dict() for product in request.products]
        products_df = pd.DataFrame(products_data)
        
        historical_data = [demand.dict() for demand in request.historical_data]
        historical_df = pd.DataFrame(historical_data)
        
        # Generate alerts
        alerts = inventory_model.generate_stock_alerts(products_df, historical_df)
        
        # Calculate summary statistics
        total_alerts = len(alerts)
        critical_alerts = len([alert for alert in alerts if alert['urgency_level'] == 'critical'])
        
        summary_by_type = {}
        for alert in alerts:
            alert_type = alert['alert_type']
            summary_by_type[alert_type] = summary_by_type.get(alert_type, 0) + 1
        
        # Generate recommended actions
        recommended_actions = []
        
        if summary_by_type.get('low_stock', 0) > 0:
            recommended_actions.append("Prioritize reordering for low stock items")
            recommended_actions.append("Review supplier lead times and reliability")
        
        if summary_by_type.get('overstock', 0) > 0:
            recommended_actions.append("Consider promotional activities for overstocked items")
            recommended_actions.append("Review demand forecasting accuracy")
        
        if summary_by_type.get('expiring', 0) > 0:
            recommended_actions.append("Implement clearance pricing for expiring inventory")
            recommended_actions.append("Review expiration date management processes")
        
        return StockAlertResponse(
            alerts=[StockAlert(**alert) for alert in alerts],
            total_alerts=total_alerts,
            critical_alerts=critical_alerts,
            summary_by_type=summary_by_type,
            recommended_actions=recommended_actions
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stock alerts error: {str(e)}")

@app.post("/generate-sample-data")
async def generate_sample_data(n_products: int = 10, n_days: int = 365):
    """Generate sample inventory data for testing"""
    try:
        # Generate products data
        products_df = inventory_model.generate_synthetic_inventory_data(n_products, n_days)
        
        # Generate historical demand data
        historical_df = inventory_model.generate_historical_demand(products_df, n_days)
        
        # Convert to response format
        products = []
        for _, row in products_df.iterrows():
            product = ProductData(
                product_id=row['product_id'],
                product_name=row['product_name'],
                category=row['category'],
                unit_cost=float(row['unit_cost']),
                holding_cost_rate=float(row['holding_cost_rate']),
                ordering_cost=float(row['ordering_cost']),
                lead_time_days=int(row['lead_time_days']),
                current_stock=int(row['current_stock']),
                demand_rate=float(row['demand_rate']),
                demand_std=float(row['demand_std']),
                service_level=float(row['service_level']),
                expiration_days=int(row['expiration_days']) if pd.notna(row['expiration_days']) else None,
                min_order_quantity=int(row['min_order_quantity']),
                max_stock_capacity=int(row['max_stock_capacity']),
                supplier_reliability=float(row['supplier_reliability'])
            )
            products.append(product)
        
        # Get historical demand
        historical_demand = []
        for _, row in historical_df.iterrows():
            demand = HistoricalDemand(
                date=row['date'].date(),
                demand=float(row['demand']),
                stock_out=bool(row['stock_out']),
                lost_sales=float(row['lost_sales']) if pd.notna(row['lost_sales']) else None
            )
            historical_demand.append(demand)
        
        return {
            "products": products,
            "historical_demand": historical_demand,
            "total_products": len(products),
            "total_demand_records": len(historical_demand),
            "summary_stats": {
                "avg_unit_cost": float(products_df['unit_cost'].mean()),
                "avg_demand_rate": float(products_df['demand_rate'].mean()),
                "avg_lead_time": float(products_df['lead_time_days'].mean()),
                "categories": products_df['category'].value_counts().to_dict()
            },
            "date_range": {
                "start": historical_df['date'].min().isoformat(),
                "end": historical_df['date'].max().isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sample data generation error: {str(e)}")

@app.post("/demand-forecast")
async def demand_forecast(historical_demand: List[HistoricalDemand], periods: int = 30):
    """Generate demand forecast for inventory planning"""
    try:
        # Extract demand values
        demand_values = [float(demand.demand) for demand in historical_demand]
        
        # Generate forecast
        forecasts = inventory_model.forecast_demand(demand_values, periods)
        
        # Create forecast dates
        last_date = max(demand.date for demand in historical_demand)
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(periods)]
        
        forecast_data = []
        for date, forecast in zip(forecast_dates, forecasts):
            forecast_data.append({
                'date': date.isoformat(),
                'predicted_demand': forecast,
                'confidence_lower': forecast * 0.8,
                'confidence_upper': forecast * 1.2
            })
        
        return {
            "forecast": forecast_data,
            "periods": periods,
            "method": "moving_average_with_trend",
            "confidence_level": 0.8
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demand forecast error: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Get information about the inventory optimization model"""
    try:
        return inventory_model.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model info error: {str(e)}")

@app.post("/calculate-metrics")
async def calculate_metrics(products: List[ProductData]):
    """Calculate inventory optimization metrics"""
    try:
        # Convert to DataFrame
        products_data = [product.dict() for product in products]
        products_df = pd.DataFrame(products_data)
        
        # Calculate metrics
        total_inventory_value = (products_df['current_stock'] * products_df['unit_cost']).sum()
        avg_service_level = products_df['service_level'].mean()
        
        # Calculate inventory turnover (simplified)
        total_annual_demand = products_df['annual_demand'].sum()
        avg_inventory_value = total_inventory_value / len(products_df)
        inventory_turnover_ratio = total_annual_demand / avg_inventory_value if avg_inventory_value > 0 else 0
        
        # Stockout rate (simplified assumption)
        stockout_rate = 0.05  # 5% default assumption
        
        # Holding cost percentage
        avg_holding_cost_rate = products_df['holding_cost_rate'].mean()
        holding_cost_percentage = avg_holding_cost_rate * 100
        
        # Ordering frequency (simplified)
        avg_ordering_cost = products_df['ordering_cost'].mean()
        ordering_frequency = 12  # Assume monthly ordering
        
        # Optimization savings (simplified estimate)
        optimization_savings = total_inventory_value * 0.1  # Assume 10% savings potential
        
        return OptimizationMetrics(
            total_inventory_value=total_inventory_value,
            average_service_level=avg_service_level,
            inventory_turnover_ratio=inventory_turnover_ratio,
            stockout_rate=stockout_rate,
            holding_cost_percentage=holding_cost_percentage,
            ordering_frequency=ordering_frequency,
            optimization_savings=optimization_savings
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics calculation error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)