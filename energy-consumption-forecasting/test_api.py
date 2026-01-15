import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def test_energy_forecasting_api():
    """Test the Energy Consumption Forecasting API"""
    
    base_url = "http://localhost:8002"
    
    print("Testing Energy Consumption Forecasting API...")
    
    # Test 1: Health check
    print("\n1. Testing health check endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("✓ Health check passed")
            print(f"Response: {response.json()}")
        else:
            print("✗ Health check failed")
    except Exception as e:
        print(f"✗ Health check error: {e}")
    
    # Test 2: Generate sample data
    print("\n2. Testing sample data generation...")
    try:
        response = requests.post(f"{base_url}/generate-sample-data", params={"days": 30, "building_type": "office"})
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            sample_data = response.json()
            print(f"✓ Generated {len(sample_data['sample_data'])} sample data points")
        else:
            print("✗ Sample data generation failed")
    except Exception as e:
        print(f"✗ Sample data generation error: {e}")
    
    # Test 3: Create historical data for forecasting
    print("\n3. Creating historical data for forecasting...")
    
    historical_data = []
    base_date = datetime.now() - timedelta(days=7)
    
    for i in range(7 * 24):
        current_time = base_date + timedelta(hours=i)
        hour = current_time.hour
        
        if 8 <= hour <= 17:
            base_consumption = 80 + np.random.normal(0, 10)
        elif 6 <= hour <= 7 or 18 <= hour <= 22:
            base_consumption = 50 + np.random.normal(0, 8)
        else:
            base_consumption = 20 + np.random.normal(0, 5)
        
        month = current_time.month
        if month in [12, 1, 2]:
            base_consumption *= 1.3
        elif month in [6, 7, 8]:
            base_consumption *= 1.2
        
        historical_data.append({
            "timestamp": current_time.isoformat(),
            "energy_consumption_kwh": max(0, base_consumption),
            "temperature_celsius": 20 + np.random.normal(0, 5),
            "humidity_percent": 60 + np.random.normal(0, 10),
            "occupancy_rate": 0.7 + np.random.normal(0, 0.2),
            "weather_condition": np.random.choice(["sunny", "cloudy", "rainy"]),
            "is_holiday": False
        })
    
    print(f"Created {len(historical_data)} historical data points")
    
    # Test 4: XGBoost Forecast
    print("\n4. Testing XGBoost forecast...")
    forecast_request = {
        "historical_data": historical_data,
        "forecast_hours": 24,
        "model_type": "xgboost",
        "building_type": "office",
        "include_confidence_intervals": True
    }
    
    try:
        response = requests.post(f"{base_url}/forecast", json=forecast_request)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            forecast_result = response.json()
            print("✓ XGBoost forecast successful")
            print(f"Total forecast consumption: {forecast_result['total_forecast_consumption']:.2f} kWh")
            print(f"Average hourly consumption: {forecast_result['average_hourly_consumption']:.2f} kWh")
            print(f"Peak consumption hour: {forecast_result['peak_consumption_hour']}")
        else:
            print(f"✗ XGBoost forecast failed: {response.text}")
    except Exception as e:
        print(f"✗ XGBoost forecast error: {e}")
    
    # Test 5: Batch Forecast
    print("\n5. Testing batch forecast...")
    batch_request = {
        "historical_data": historical_data,
        "forecast_hours": 48,
        "model_types": ["xgboost"],
        "building_type": "office"
    }
    
    try:
        response = requests.post(f"{base_url}/forecast/batch", json=batch_request)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            batch_result = response.json()
            print("✓ Batch forecast successful")
            for model_type, result in batch_result['forecasts'].items():
                print(f"  {model_type}: {len(result['forecast'])} points, total: {result['total_forecast_consumption']:.2f} kWh")
        else:
            print(f"✗ Batch forecast failed: {response.text}")
    except Exception as e:
        print(f"✗ Batch forecast error: {e}")
    
    print("\n" + "="*50)
    print("Energy Consumption Forecasting API testing completed!")
    print("="*50)

if __name__ == "__main__":
    test_energy_forecasting_api()