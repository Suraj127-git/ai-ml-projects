"""
Test script for Demand Forecasting with Neural Networks API
"""
import requests
import json
from datetime import datetime, timedelta
import random

# API base URL
BASE_URL = "http://localhost:8003"

def generate_sample_historical_data(n_days: int = 100) -> list:
    """Generate sample historical demand data"""
    base_date = datetime.now() - timedelta(days=n_days)
    data = []
    
    for i in range(n_days):
        date = base_date + timedelta(days=i)
        # Generate demand with some seasonality and randomness
        base_demand = 100 + 20 * (i % 7)  # Weekly pattern
        seasonal_demand = 30 * (i % 30)  # Monthly pattern
        noise = random.randint(-20, 20)
        demand = max(0, base_demand + seasonal_demand + noise)
        
        data.append({
            "date": date.isoformat(),
            "demand": float(demand)
        })
    
    return data

def test_api_endpoints():
    """Test all API endpoints"""
    print("Testing Demand Forecasting Neural Networks API...")
    
    # Test health check
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health Check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Health Check Failed: {e}")
    
    # Test root endpoint
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Root Endpoint: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Root Endpoint Failed: {e}")
    
    # Test sample data generation
    print("\nTesting sample data generation...")
    try:
        response = requests.post(f"{BASE_URL}/generate-sample-data", params={"n_days": 100})
        sample_data = response.json()
        print(f"Sample Data Generation: {response.status_code}")
        if response.status_code == 200:
            print(f"Generated {len(sample_data.get('sample_data', []))} days of sample data")
            if sample_data.get('sample_data'):
                print(f"First data point: {sample_data['sample_data'][0]}")
    except Exception as e:
        print(f"Sample Data Generation Failed: {e}")
    
    # Test model training for each model type
    model_types = ["lstm", "gru", "transformer", "cnn_lstm", "bilstm"]
    
    for model_type in model_types:
        print(f"\nTraining {model_type.upper()} model...")
        try:
            response = requests.post(f"{BASE_URL}/train/{model_type}", params={"n_days": 500})
            training_result = response.json()
            print(f"{model_type.upper()} Training: {response.status_code}")
            if response.status_code == 200:
                print(f"Training completed: {training_result.get('message')}")
                performance = training_result.get('model_performance', {})
                print(f"RMSE: {performance.get('rmse', 0):.2f}, MAE: {performance.get('mae', 0):.2f}")
        except Exception as e:
            print(f"{model_type.upper()} Training Failed: {e}")
    
    # Test forecasting with LSTM model
    print("\nTesting demand forecasting...")
    historical_data = generate_sample_historical_data(50)
    
    forecast_request = {
        "product_id": "PROD_TEST_001",
        "model_type": "lstm",
        "historical_data": historical_data,
        "forecast_horizon": 14
    }
    
    try:
        response = requests.post(f"{BASE_URL}/forecast", json=forecast_request)
        forecast_result = response.json()
        print(f"LSTM Forecast: {response.status_code}")
        if response.status_code == 200:
            forecasts = forecast_result.get('forecasts', [])
            print(f"Forecasted {len(forecasts)} days")
            if forecasts:
                print(f"First forecast: {forecasts[0]}")
                print(f"Last forecast: {forecasts[-1]}")
            performance = forecast_result.get('performance_metrics', {})
            print(f"Performance - RMSE: {performance.get('rmse', 0):.2f}, MAE: {performance.get('mae', 0):.2f}")
    except Exception as e:
        print(f"LSTM Forecast Failed: {e}")
    
    # Test forecasting with GRU model
    print("\nTesting GRU forecasting...")
    forecast_request["model_type"] = "gru"
    
    try:
        response = requests.post(f"{BASE_URL}/forecast", json=forecast_request)
        forecast_result = response.json()
        print(f"GRU Forecast: {response.status_code}")
        if response.status_code == 200:
            forecasts = forecast_result.get('forecasts', [])
            print(f"Forecasted {len(forecasts)} days")
    except Exception as e:
        print(f"GRU Forecast Failed: {e}")
    
    # Test model info
    print("\nTesting model info...")
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        model_info = response.json()
        print(f"Model Info: {response.status_code}")
        if response.status_code == 200:
            available_models = model_info.get('available_models', [])
            print(f"Available models: {available_models}")
            print(f"Device: {model_info.get('device', 'unknown')}")
            performance = model_info.get('model_performance', {})
            for model_type, perf in performance.items():
                print(f"{model_type.upper()}: RMSE={perf.get('rmse', 0):.2f}, MAE={perf.get('mae', 0):.2f}")
    except Exception as e:
        print(f"Model Info Failed: {e}")
    
    # Test model performance
    print("\nTesting model performance...")
    try:
        response = requests.get(f"{BASE_URL}/model/performance")
        performance = response.json()
        print(f"Model Performance: {response.status_code}")
        if response.status_code == 200:
            print(f"Models trained: {performance.get('available_models', [])}")
            print(f"Performance metrics: {performance.get('model_performance', {})}")
    except Exception as e:
        print(f"Model Performance Failed: {e}")
    
    print("\nAPI testing completed!")

if __name__ == "__main__":
    print("Make sure the Demand Forecasting Neural Networks API is running on http://localhost:8003")
    print("You can start it with: python -m demand-forecasting-neural.app.main")
    print()
    test_api_endpoints()