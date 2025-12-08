import requests
import json
import numpy as np
import time
from datetime import datetime, timedelta

def create_test_product_data():
    """Create test product data for price optimization"""
    test_products = [
        {
            "product_id": "laptop_001",
            "product_name": "Gaming Laptop Pro",
            "category": "electronics",
            "current_price": 1299.99,
            "cost_price": 800.00,
            "competitor_prices": [1199.99, 1349.99, 1275.00, 1325.50],
            "demand_history": [45, 52, 38, 41, 48, 55, 42],
            "inventory_level": 25,
            "seasonality_factor": 1.2,
            "price_elasticity": -1.5,
            "target_margin": 0.25,
            "market_conditions": "high_demand"
        },
        {
            "product_id": "shirt_002",
            "product_name": "Designer Cotton Shirt",
            "category": "clothing",
            "current_price": 79.99,
            "cost_price": 35.00,
            "competitor_prices": [69.99, 89.99, 75.00, 82.50],
            "demand_history": [120, 135, 98, 110, 125, 140, 105],
            "inventory_level": 200,
            "seasonality_factor": 0.9,
            "price_elasticity": -2.0,
            "target_margin": 0.35,
            "market_conditions": "normal"
        },
        {
            "product_id": "coffee_003",
            "product_name": "Premium Coffee Beans",
            "category": "food",
            "current_price": 24.99,
            "cost_price": 12.00,
            "competitor_prices": [22.99, 27.99, 25.50, 23.75],
            "demand_history": [200, 180, 220, 195, 210, 175, 205],
            "inventory_level": 500,
            "seasonality_factor": 1.1,
            "price_elasticity": -1.8,
            "target_margin": 0.40,
            "market_conditions": "seasonal"
        }
    ]
    
    return test_products

def test_health_check():
    """Test health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get("http://localhost:8009/health")
        print(f"Health check status: {response.status_code}")
        print(f"Health check response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_single_price_optimization():
    """Test single price optimization"""
    print("\nTesting single price optimization...")
    
    try:
        test_products = create_test_product_data()
        product = test_products[0]  # Use laptop as test
        
        # Send optimization request
        response = requests.post(
            "http://localhost:8009/optimize-price",
            json=product
        )
        
        print(f"Price optimization status: {response.status_code}")
        result = response.json()
        
        if response.status_code == 200:
            print(f"Product: {result['product_id']}")
            print(f"Current Price: ${result['current_price']:.2f}")
            print(f"Optimized Price: ${result['optimized_price']:.2f}")
            print(f"Expected Revenue: ${result['expected_revenue']:.2f}")
            print(f"Expected Demand: {result['expected_demand']:.0f} units")
            print(f"Profit Margin: {result['profit_margin']:.1%}")
            print(f"Confidence: {result['confidence_score']:.2f}")
            print(f"Reasoning: {result['reasoning']}")
            print(f"Strategy: {result['strategy_used']}")
            return True
        else:
            print(f"Price optimization error: {result}")
            return False
            
    except Exception as e:
        print(f"Price optimization test failed: {e}")
        return False

def test_batch_price_optimization():
    """Test batch price optimization"""
    print("\nTesting batch price optimization...")
    
    try:
        test_products = create_test_product_data()
        
        # Prepare batch request
        batch_request = {
            "products": test_products,
            "optimization_strategy": "reinforcement_learning",
            "max_price_change": 0.2,
            "business_constraints": {
                "min_margin": 0.15,
                "max_price_increase": 0.25
            }
        }
        
        # Send batch optimization request
        response = requests.post(
            "http://localhost:8009/optimize-batch",
            json=batch_request
        )
        
        print(f"Batch optimization status: {response.status_code}")
        result = response.json()
        
        if response.status_code == 200:
            print(f"Total products: {len(result['products'])}")
            print(f"Total expected revenue: ${result['total_expected_revenue']:.2f}")
            print(f"Optimization time: {result['optimization_time']:.3f}s")
            print(f"Strategy used: {result['strategy_used']}")
            
            # Show first product result
            if result['products']:
                first_product = result['products'][0]
                print(f"First product: {first_product['product_id']}")
                print(f"  Current: ${first_product['current_price']:.2f} → Optimized: ${first_product['optimized_price']:.2f}")
            
            if result['constraint_violations']:
                print(f"Constraint violations: {len(result['constraint_violations'])}")
                for violation in result['constraint_violations'][:2]:  # Show first 2
                    print(f"  - {violation}")
            
            return True
        else:
            print(f"Batch optimization error: {result}")
            return False
            
    except Exception as e:
        print(f"Batch optimization test failed: {e}")
        return False

def test_price_elasticity_analysis():
    """Test price elasticity analysis"""
    print("\nTesting price elasticity analysis...")
    
    try:
        # Create price and demand history
        price_history = [100, 105, 110, 95, 90, 85, 95, 100]
        demand_history = [100, 90, 80, 120, 130, 140, 115, 105]
        
        # Send elasticity analysis request
        response = requests.post(
            "http://localhost:8009/analyze-elasticity",
            params={"product_id": "test_product"},
            json={"price_history": price_history, "demand_history": demand_history}
        )
        
        print(f"Elasticity analysis status: {response.status_code}")
        result = response.json()
        
        if response.status_code == 200:
            print(f"Product: {result['product_id']}")
            print(f"Elasticity coefficient: {result['elasticity_coefficient']:.3f}")
            print(f"Elasticity category: {result['elasticity_category']}")
            print(f"Confidence interval: [{result['confidence_interval'][0]:.3f}, {result['confidence_interval'][1]:.3f}]")
            print(f"Statistical significance: {result['statistical_significance']:.3f}")
            return True
        else:
            print(f"Elasticity analysis error: {result}")
            return False
            
    except Exception as e:
        print(f"Elasticity analysis test failed: {e}")
        return False

def test_competitive_analysis():
    """Test competitive analysis"""
    print("\nTesting competitive analysis...")
    
    try:
        product_price = 1299.99
        competitor_prices = [1199.99, 1349.99, 1275.00, 1325.50, 1289.99]
        
        # Send competitive analysis request
        response = requests.post(
            "http://localhost:8009/analyze-competition",
            params={
                "product_id": "laptop_001",
                "product_price": product_price
            },
            json=competitor_prices
        )
        
        print(f"Competitive analysis status: {response.status_code}")
        result = response.json()
        
        if response.status_code == 200:
            print(f"Product: {result['product_id']}")
            print(f"Price position: {result['price_position']}")
            print(f"Competitive advantage: {result['competitive_advantage']:.3f}")
            print(f"Market pricing trend: {result['market_pricing_trend']}")
            print(f"Competitor prices: {result['competitor_prices']}")
            return True
        else:
            print(f"Competitive analysis error: {result}")
            return False
            
    except Exception as e:
        print(f"Competitive analysis test failed: {e}")
        return False

def test_pricing_strategies():
    """Test different pricing strategies"""
    print("\nTesting different pricing strategies...")
    
    try:
        test_products = create_test_product_data()
        product = test_products[1]  # Use shirt as test
        
        strategies = ["reinforcement_learning", "dynamic_pricing", "competitive_pricing"]
        results = {}
        
        for strategy in strategies:
            print(f"\nTesting {strategy}...")
            
            response = requests.post(
                f"http://localhost:8009/optimize-price/{strategy}",
                json=product
            )
            
            if response.status_code == 200:
                result = response.json()
                results[strategy] = {
                    "optimized_price": result['optimized_price'],
                    "expected_revenue": result['expected_revenue'],
                    "strategy_used": result['strategy_used']
                }
                print(f"  Optimized price: ${result['optimized_price']:.2f}")
                print(f"  Expected revenue: ${result['expected_revenue']:.2f}")
            else:
                print(f"  Failed: {response.text}")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"Pricing strategies test failed: {e}")
        return False

def test_generate_sample_data():
    """Test sample data generation"""
    print("\nTesting sample data generation...")
    
    try:
        response = requests.get(
            "http://localhost:8009/generate-sample-data",
            params={"n_samples": 50}
        )
        
        print(f"Sample data generation status: {response.status_code}")
        result = response.json()
        
        if response.status_code == 200:
            print(f"Generated {result['count']} samples")
            print(f"Categories: {result['categories']}")
            print(f"Sample data preview: {len(result['sample_data'])} records")
            return True
        else:
            print(f"Sample data generation error: {result}")
            return False
            
    except Exception as e:
        print(f"Sample data generation test failed: {e}")
        return False

def test_model_performance():
    """Test model performance endpoint"""
    print("\nTesting model performance...")
    
    try:
        response = requests.get("http://localhost:8009/model-performance")
        
        print(f"Model performance status: {response.status_code}")
        result = response.json()
        
        if response.status_code == 200:
            print("Models status:")
            for model_name, status in result['models_status'].items():
                print(f"  {model_name}: {status}")
            
            print("Training history:")
            for key, value in result['training_history'].items():
                print(f"  {key}: {value}")
            
            return True
        else:
            print(f"Model performance error: {result}")
            return False
            
    except Exception as e:
        print(f"Model performance test failed: {e}")
        return False

def test_pricing_strategies_list():
    """Test pricing strategies list endpoint"""
    print("\nTesting pricing strategies list...")
    
    try:
        response = requests.get("http://localhost:8009/pricing-strategies")
        
        print(f"Pricing strategies status: {response.status_code}")
        result = response.json()
        
        if response.status_code == 200:
            print("Available strategies:")
            for strategy in result['strategies']:
                print(f"  - {strategy['name']}: {strategy['description']}")
            return True
        else:
            print(f"Pricing strategies error: {result}")
            return False
            
    except Exception as e:
        print(f"Pricing strategies list test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("Starting Price Optimization Engine API Tests")
    print("=" * 60)
    
    tests = [
        ("Health Check", test_health_check),
        ("Single Price Optimization", test_single_price_optimization),
        ("Batch Price Optimization", test_batch_price_optimization),
        ("Price Elasticity Analysis", test_price_elasticity_analysis),
        ("Competitive Analysis", test_competitive_analysis),
        ("Pricing Strategies", test_pricing_strategies),
        ("Sample Data Generation", test_generate_sample_data),
        ("Model Performance", test_model_performance),
        ("Pricing Strategies List", test_pricing_strategies_list)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"{test_name}: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            print(f"{test_name}: FAILED - {e}")
            results.append((test_name, False))
        
        # Small delay between tests
        time.sleep(0.5)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        print(f"{test_name}: {'✓' if success else '✗'}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {total - passed} tests failed")
    
    return passed == total

if __name__ == "__main__":
    # Wait a moment for the server to be ready
    print("Waiting for server to be ready...")
    time.sleep(2)
    
    success = run_all_tests()
    exit(0 if success else 1)