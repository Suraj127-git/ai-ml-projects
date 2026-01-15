"""
Sales Forecasting API Test Suite
Tests for the Prophet-based sales forecasting system
"""

import requests
import json
import time
from datetime import datetime, timedelta

# API base URL
BASE_URL = "http://localhost:8001"

def test_health_check():
    """Test health check endpoint"""
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Health check passed: {data['status']}")
        print(f"  Available models: {data.get('available_models', [])}")
        print(f"  Loaded models: {data.get('loaded_models', 0)}")
        return True
    else:
        print(f"✗ Health check failed: {response.status_code}")
        return False

def test_root_endpoint():
    """Test root endpoint"""
    print("\nTesting root endpoint...")
    response = requests.get(f"{BASE_URL}/")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Root endpoint working: {data['message']}")
        print(f"  Version: {data['version']}")
        print(f"  Available models: {data.get('available_models', [])}")
        return True
    else:
        print(f"✗ Root endpoint failed: {response.status_code}")
        return False

def test_generate_sample_data():
    """Test sample data generation"""
    print("\nTesting sample data generation...")
    
    payload = {
        "days": 365
    }
    
    response = requests.post(f"{BASE_URL}/generate-sample-data", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Sample data generated: {data['count']} records")
        
        # Save sample data for testing
        with open("sample_sales_data.json", "w") as f:
            json.dump(data['sample_data'][:10], f, indent=2)  # Save first 10 records
        
        print(f"  Sample saved to sample_sales_data.json")
        return data['sample_data']
    else:
        print(f"✗ Sample data generation failed: {response.status_code}")
        print(f"  Response: {response.text}")
        return None

def test_train_models():
    """Test model training"""
    print("\nTesting model training...")
    
    models = ['prophet', 'arima', 'linear_regression']
    results = []
    
    for model_type in models:
        print(f"  Training {model_type}...")
        
        response = requests.post(f"{BASE_URL}/train/{model_type}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ {model_type} model trained: {data['message']}")
            results.append(True)
        else:
            print(f"  ✗ {model_type} training failed: {response.status_code}")
            print(f"    Response: {response.text}")
            results.append(False)
        
        # Wait a bit between trainings
        time.sleep(2)
    
    return all(results)

def test_forecast_single():
    """Test single forecast"""
    print("\nTesting single forecast...")
    
    # Get sample data first
    sample_data = test_generate_sample_data()
    if not sample_data:
        return False
    
    # Use last 30 days as historical data
    historical_data = sample_data[-30:]
    
    payload = {
        "historical_data": historical_data,
        "model_type": "prophet",
        "forecast_days": 7
    }
    
    response = requests.post(f"{BASE_URL}/forecast", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        forecast = data['forecast']
        print(f"✓ Single forecast successful: {len(forecast)} days forecasted")
        
        # Show first few predictions
        for i, pred in enumerate(forecast[:3]):
            print(f"  Day {i+1}: {pred['date']} - Sales: {pred['sales']:.2f}")
        
        return True
    else:
        print(f"✗ Single forecast failed: {response.status_code}")
        print(f"  Response: {response.text}")
        return False

def test_forecast_batch():
    """Test batch forecast with multiple models"""
    print("\nTesting batch forecast...")
    
    # Get sample data
    sample_data = test_generate_sample_data()
    if not sample_data:
        return False
    
    models = ['prophet', 'arima', 'linear_regression']
    results = []
    
    for model_type in models:
        print(f"  Testing {model_type}...")
        
        # Use last 30 days as historical data
        historical_data = sample_data[-30:]
        
        payload = {
            "historical_data": historical_data,
            "model_type": model_type,
            "forecast_days": 14
        }
        
        response = requests.post(f"{BASE_URL}/forecast", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            forecast = data['forecast']
            print(f"    ✓ {model_type}: {len(forecast)} days forecasted")
            results.append(True)
        else:
            print(f"    ✗ {model_type} forecast failed: {response.status_code}")
            results.append(False)
    
    return all(results)

def test_model_info():
    """Test model information endpoints"""
    print("\nTesting model information...")
    
    # Test available models
    response = requests.get(f"{BASE_URL}/models")
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Available models: {data['available_models']}")
        print(f"  Loaded models: {data['loaded_models']}")
    
    # Test specific model info
    models = ['prophet', 'arima', 'linear_regression']
    results = []
    
    for model_type in models:
        response = requests.get(f"{BASE_URL}/model-info/{model_type}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ {model_type} info retrieved")
            print(f"    Type: {data['model_type']}")
            print(f"    Training date: {data['training_date']}")
            results.append(True)
        else:
            print(f"  ✗ {model_type} info failed: {response.status_code}")
            results.append(False)
    
    return all(results)

def test_error_handling():
    """Test error handling"""
    print("\nTesting error handling...")
    
    results = []
    
    # Test invalid model type
    payload = {
        "historical_data": [],
        "model_type": "invalid_model",
        "forecast_days": 7
    }
    
    response = requests.post(f"{BASE_URL}/forecast", json=payload)
    if response.status_code == 400:
        print("✓ Invalid model type properly rejected")
        results.append(True)
    else:
        print("✗ Invalid model type not properly handled")
        results.append(False)
    
    # Test empty historical data
    payload = {
        "historical_data": [],
        "model_type": "prophet",
        "forecast_days": 7
    }
    
    response = requests.post(f"{BASE_URL}/forecast", json=payload)
    if response.status_code in [400, 500]:
        print("✓ Empty historical data properly handled")
        results.append(True)
    else:
        print("✗ Empty historical data not properly handled")
        results.append(False)
    
    return all(results)

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("SALES FORECASTING API TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_health_check,
        test_root_endpoint,
        test_generate_sample_data,
        test_train_models,
        test_forecast_single,
        test_forecast_batch,
        test_model_info,
        test_error_handling
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ ALL TESTS PASSED - API is working correctly!")
    else:
        print("✗ SOME TESTS FAILED - Please check the issues above")
    
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    # Wait a moment for the server to be ready
    print("Waiting for API server to be ready...")
    time.sleep(3)
    
    success = run_all_tests()
    exit(0 if success else 1)