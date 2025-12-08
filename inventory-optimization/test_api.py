"""
Test script for Inventory Optimization API
"""
import requests
import json
from datetime import datetime, timedelta
import numpy as np

# API base URL
BASE_URL = "http://localhost:8004"

def generate_sample_product_data(product_id="TEST_PRODUCT_001"):
    """Generate sample product data for testing"""
    return {
        "product_id": product_id,
        "product_name": f"Test Product {product_id.split('_')[-1]}",
        "category": "Electronics",
        "unit_cost": 50.0,
        "holding_cost_rate": 0.25,
        "ordering_cost": 100.0,
        "lead_time_days": 14,
        "current_stock": 200,
        "demand_rate": 10.0,
        "demand_std": 2.0,
        "service_level": 0.95,
        "expiration_days": 365,
        "min_order_quantity": 1,
        "max_stock_capacity": 1000,
        "supplier_reliability": 0.95
    }

def generate_sample_historical_demand(product_id="TEST_PRODUCT_001", days=30):
    """Generate sample historical demand data"""
    historical_data = []
    base_demand = 10.0
    
    for i in range(days):
        date = datetime.now() - timedelta(days=days-i)
        # Add some randomness to demand
        demand = max(0, np.random.normal(base_demand, 2.0))
        stock_out = np.random.random() < 0.05  # 5% chance of stockout
        lost_sales = demand * 0.3 if stock_out else 0
        
        historical_data.append({
            "date": date.strftime("%Y-%m-%d"),
            "demand": demand,
            "stock_out": stock_out,
            "lost_sales": lost_sales if lost_sales > 0 else None
        })
    
    return historical_data

def test_api_endpoints():
    """Test all API endpoints"""
    print("Testing Inventory Optimization API...")
    
    # Test health check
    try:
        response = requests.get(f"{BASE_URL}/health")
        health_data = response.json()
        print(f"Health Check: {response.status_code}")
        print(f"Status: {health_data.get('status')}")
        print(f"Available methods: {health_data.get('available_methods')}")
    except Exception as e:
        print(f"Health Check Failed: {e}")
    
    # Test root endpoint
    try:
        response = requests.get(f"{BASE_URL}/")
        root_data = response.json()
        print(f"\nRoot Endpoint: {response.status_code}")
        print(f"Message: {root_data.get('message')}")
        print(f"Available methods: {root_data.get('available_methods')}")
    except Exception as e:
        print(f"Root Endpoint Failed: {e}")
    
    # Test sample data generation
    try:
        response = requests.post(f"{BASE_URL}/generate-sample-data", params={"n_products": 5, "n_days": 30})
        sample_data = response.json()
        print(f"\nSample Data Generation: {response.status_code}")
        print(f"Generated {sample_data.get('total_products')} products")
        print(f"Generated {sample_data.get('total_demand_records')} demand records")
        
        # Store sample data for later use
        if sample_data.get('products'):
            sample_products = sample_data['products']
            print(f"Sample product categories: {list(set(p.get('category') for p in sample_products))}")
    except Exception as e:
        print(f"Sample Data Generation Failed: {e}")
    
    # Test EOQ optimization
    print("\nTesting EOQ optimization...")
    sample_product = generate_sample_product_data("EOQ_TEST_001")
    
    optimization_request = {
        "product_data": sample_product,
        "optimization_method": "eoq",
        "service_level": 0.95,
        "forecast_horizon_days": 30
    }
    
    try:
        response = requests.post(f"{BASE_URL}/optimize", json=optimization_request)
        optimization_result = response.json()
        print(f"EOQ Optimization: {response.status_code}")
        
        if response.status_code == 200:
            print(f"Economic Order Quantity: {optimization_result.get('economic_order_quantity')}")
            print(f"Reorder Point: {optimization_result.get('reorder_point')}")
            print(f"Safety Stock: {optimization_result.get('safety_stock')}")
            print(f"Total Cost: ${optimization_result.get('total_cost'):.2f}")
            print(f"Ordering Cost: ${optimization_result.get('ordering_cost'):.2f}")
            print(f"Holding Cost: ${optimization_result.get('holding_cost'):.2f}")
            print(f"Service Level Achieved: {optimization_result.get('service_level_achieved'):.2%}")
            print(f"Inventory Turnover: {optimization_result.get('inventory_turnover'):.2f}")
            print(f"Days of Supply: {optimization_result.get('days_of_supply')}")
            
            recommendations = optimization_result.get('recommendations', [])
            if recommendations:
                print(f"Recommendations:")
                for rec in recommendations:
                    print(f"  - {rec}")
    except Exception as e:
        print(f"EOQ Optimization Failed: {e}")
    
    # Test safety stock optimization
    print("\nTesting Safety Stock optimization...")
    safety_request = {
        "product_data": sample_product,
        "optimization_method": "safety_stock",
        "service_level": 0.99
    }
    
    try:
        response = requests.post(f"{BASE_URL}/optimize", json=safety_request)
        safety_result = response.json()
        print(f"Safety Stock Optimization: {response.status_code}")
        
        if response.status_code == 200:
            print(f"Safety Stock: {safety_result.get('safety_stock')}")
            print(f"Reorder Point: {safety_result.get('reorder_point')}")
            print(f"Service Level: {safety_result.get('service_level_achieved'):.2%}")
    except Exception as e:
        print(f"Safety Stock Optimization Failed: {e}")
    
    # Test batch optimization
    print("\nTesting batch optimization...")
    batch_products = [
        {
            "product_data": generate_sample_product_data("BATCH_001"),
            "optimization_method": "eoq",
            "service_level": 0.95
        },
        {
            "product_data": generate_sample_product_data("BATCH_002"),
            "optimization_method": "eoq",
            "service_level": 0.98
        }
    ]
    
    batch_request = {
        "products": batch_products,
        "optimization_method": "eoq"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/optimize/batch", json=batch_request)
        batch_result = response.json()
        print(f"Batch Optimization: {response.status_code}")
        
        if response.status_code == 200:
            print(f"Optimized {batch_result.get('total_products')} products")
            print(f"Processing time: {batch_result.get('processing_time'):.2f} seconds")
            
            summary_stats = batch_result.get('summary_stats', {})
            if summary_stats:
                print(f"Total Inventory Value: ${summary_stats.get('total_inventory_value'):.2f}")
                print(f"Average Service Level: {summary_stats.get('average_service_level'):.2%}")
                print(f"Total Annual Cost: ${summary_stats.get('total_annual_cost'):.2f}")
            
            results = batch_result.get('results', [])
            for i, result in enumerate(results):
                print(f"Product {i+1}: EOQ={result.get('economic_order_quantity')}, "
                      f"ROP={result.get('reorder_point')}, "
                      f"Cost=${result.get('total_cost'):.2f}")
    except Exception as e:
        print(f"Batch Optimization Failed: {e}")
    
    # Test ABC Analysis
    print("\nTesting ABC Analysis...")
    
    # Generate multiple products for ABC analysis
    abc_products = [
        generate_sample_product_data("ABC_A_001"),
        generate_sample_product_data("ABC_A_002"),
        generate_sample_product_data("ABC_B_001"),
        generate_sample_product_data("ABC_B_002"),
        generate_sample_product_data("ABC_C_001")
    ]
    
    # Modify some products to create different revenue levels
    abc_products[0]["unit_cost"] = 200.0  # High value product A
    abc_products[0]["demand_rate"] = 50.0  # High demand
    abc_products[2]["unit_cost"] = 80.0   # Medium value product B
    abc_products[2]["demand_rate"] = 20.0  # Medium demand
    abc_products[4]["unit_cost"] = 20.0   # Low value product C
    abc_products[4]["demand_rate"] = 5.0   # Low demand
    
    # Generate historical demand for ABC analysis
    abc_historical = []
    for product in abc_products:
        product_historical = generate_sample_historical_demand(product["product_id"], 30)
        abc_historical.extend(product_historical)
    
    abc_request = {
        "products": abc_products,
        "historical_data": abc_historical,
        "revenue_percentage_a": 0.8,
        "item_percentage_a": 0.2
    }
    
    try:
        response = requests.post(f"{BASE_URL}/abc-analysis", json=abc_request)
        abc_result = response.json()
        print(f"ABC Analysis: {response.status_code}")
        
        if response.status_code == 200:
            print(f"Total Revenue: ${abc_result.get('total_revenue'):.2f}")
            
            category_summary = abc_result.get('category_summary', {})
            print(f"Category Summary:")
            for category, count in category_summary.items():
                print(f"  Category {category}: {count} products")
            
            recommendations = abc_result.get('recommendations', [])
            if recommendations:
                print(f"Recommendations:")
                for rec in recommendations:
                    print(f"  - {rec}")
            
            # Show some analysis results
            analysis_results = abc_result.get('analysis_results', [])
            if analysis_results:
                print(f"Sample Analysis Results:")
                for result in analysis_results[:3]:  # Show first 3 results
                    print(f"  {result.get('product_id')}: Category {result.get('category')}, "
                          f"Revenue: ${result.get('annual_revenue'):.2f}, "
                          f"Priority: {result.get('optimization_priority')}")
    except Exception as e:
        print(f"ABC Analysis Failed: {e}")
    
    # Test Stock Alerts
    print("\nTesting Stock Alerts...")
    
    # Create products with different stock levels to trigger alerts
    alert_products = [
        generate_sample_product_data("ALERT_LOW_001"),
        generate_sample_product_data("ALERT_NORMAL_001"),
        generate_sample_product_data("ALERT_HIGH_001")
    ]
    
    # Modify stock levels to trigger different alerts
    alert_products[0]["current_stock"] = 5    # Very low stock
    alert_products[1]["current_stock"] = 200   # Normal stock
    alert_products[2]["current_stock"] = 800   # High stock (potential overstock)
    
    alert_historical = []
    for product in alert_products:
        product_historical = generate_sample_historical_demand(product["product_id"], 30)
        alert_historical.extend(product_historical)
    
    alert_request = {
        "products": alert_products,
        "historical_data": alert_historical
    }
    
    try:
        response = requests.post(f"{BASE_URL}/stock-alerts", json=alert_request)
        alert_result = response.json()
        print(f"Stock Alerts: {response.status_code}")
        
        if response.status_code == 200:
            print(f"Total Alerts: {alert_result.get('total_alerts')}")
            print(f"Critical Alerts: {alert_result.get('critical_alerts')}")
            
            summary_by_type = alert_result.get('summary_by_type', {})
            print(f"Alerts by Type:")
            for alert_type, count in summary_by_type.items():
                print(f"  {alert_type}: {count}")
            
            alerts = alert_result.get('alerts', [])
            if alerts:
                print(f"Sample Alerts:")
                for alert in alerts[:3]:  # Show first 3 alerts
                    print(f"  {alert.get('product_id')}: {alert.get('alert_type')} "
                          f"({alert.get('urgency_level')} urgency) - {alert.get('recommended_action')}")
            
            recommended_actions = alert_result.get('recommended_actions', [])
            if recommended_actions:
                print(f"Recommended Actions:")
                for action in recommended_actions:
                    print(f"  - {action}")
    except Exception as e:
        print(f"Stock Alerts Failed: {e}")
    
    # Test Model Info
    print("\nTesting Model Info...")
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        model_info = response.json()
        print(f"Model Info: {response.status_code}")
        
        if response.status_code == 200:
            print(f"ABC Categories Count: {model_info.get('abc_categories_count')}")
            print(f"Demand Forecasters Count: {model_info.get('demand_forecasters_count')}")
            print(f"Optimization History Length: {model_info.get('optimization_history_length')}")
            print(f"Available Methods: {model_info.get('available_methods')}")
            print(f"Dependencies: {model_info.get('dependencies')}")
    except Exception as e:
        print(f"Model Info Failed: {e}")
    
    # Test Metrics Calculation
    print("\nTesting Metrics Calculation...")
    metrics_products = [
        generate_sample_product_data("METRIC_001"),
        generate_sample_product_data("METRIC_002"),
        generate_sample_product_data("METRIC_003")
    ]
    
    try:
        response = requests.post(f"{BASE_URL}/calculate-metrics", json=metrics_products)
        metrics_result = response.json()
        print(f"Metrics Calculation: {response.status_code}")
        
        if response.status_code == 200:
            print(f"Total Inventory Value: ${metrics_result.get('total_inventory_value'):.2f}")
            print(f"Average Service Level: {metrics_result.get('average_service_level'):.2%}")
            print(f"Inventory Turnover Ratio: {metrics_result.get('inventory_turnover_ratio'):.2f}")
            print(f"Stockout Rate: {metrics_result.get('stockout_rate'):.2%}")
            print(f"Holding Cost Percentage: {metrics_result.get('holding_cost_percentage'):.2f}%")
            print(f"Ordering Frequency: {metrics_result.get('ordering_frequency')} per year")
            print(f"Optimization Savings Potential: ${metrics_result.get('optimization_savings'):.2f}")
    except Exception as e:
        print(f"Metrics Calculation Failed: {e}")
    
    print("\nAPI testing completed!")

if __name__ == "__main__":
    print("Make sure the Inventory Optimization API is running on http://localhost:8004")
    print("You can start it with: python -m inventory-optimization.app.main")
    print()
    test_api_endpoints()