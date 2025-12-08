"""
Test script for CLV Predictor API
"""
import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8002"

def test_api_endpoints():
    """Test all API endpoints"""
    print("Testing CLV Predictor API...")
    
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
    try:
        response = requests.post(f"{BASE_URL}/generate-sample-data", params={"n_customers": 5})
        sample_data = response.json()
        print(f"Sample Data Generation: {response.status_code}")
        if response.status_code == 200:
            customers = sample_data.get("sample_customers", [])
            print(f"Generated {len(customers)} sample customers")
            if customers:
                print(f"First customer: {customers[0]}")
    except Exception as e:
        print(f"Sample Data Generation Failed: {e}")
    
    # Test model training (XGBoost)
    print("\nTraining XGBoost model...")
    try:
        response = requests.post(f"{BASE_URL}/train/xgboost", params={"n_samples": 500})
        training_result = response.json()
        print(f"XGBoost Training: {response.status_code}")
        if response.status_code == 200:
            print(f"Training completed: {training_result.get('message')}")
            print(f"Model performance: {training_result.get('model_performance', {})}")
    except Exception as e:
        print(f"XGBoost Training Failed: {e}")
    
    # Test model training (BG-NBD)
    print("\nTraining BG-NBD model...")
    try:
        response = requests.post(f"{BASE_URL}/train/bg_nbd", params={"n_samples": 500})
        training_result = response.json()
        print(f"BG-NBD Training: {response.status_code}")
        if response.status_code == 200:
            print(f"Training completed: {training_result.get('message')}")
    except Exception as e:
        print(f"BG-NBD Training Failed: {e}")
    
    # Test prediction with sample data
    print("\nTesting CLV prediction...")
    sample_customer = {
        "customer_id": "TEST_CUSTOMER_001",
        "recency": 30,
        "frequency": 5,
        "monetary_value": 1500.0,
        "tenure": 180,
        "avg_order_value": 300.0,
        "days_between_purchases": 36.0,
        "total_orders": 5,
        "age": 35,
        "gender": "M",
        "country": "USA"
    }
    
    # Test XGBoost prediction
    try:
        response = requests.post(f"{BASE_URL}/predict?model_type=xgboost", json=sample_customer)
        prediction = response.json()
        print(f"XGBoost Prediction: {response.status_code}")
        if response.status_code == 200:
            print(f"Predicted CLV: ${prediction.get('predicted_clv', 0):.2f}")
            print(f"Model used: {prediction.get('model_used')}")
            print(f"Confidence: {prediction.get('confidence_score', 0):.2f}")
    except Exception as e:
        print(f"XGBoost Prediction Failed: {e}")
    
    # Test BG-NBD prediction
    try:
        response = requests.post(f"{BASE_URL}/predict?model_type=bg_nbd", json=sample_customer)
        prediction = response.json()
        print(f"BG-NBD Prediction: {response.status_code}")
        if response.status_code == 200:
            print(f"Predicted CLV: ${prediction.get('predicted_clv', 0):.2f}")
            print(f"Expected purchases next 30 days: {prediction.get('expected_purchases_next_30_days', 0):.2f}")
            print(f"Expected purchases next 90 days: {prediction.get('expected_purchases_next_90_days', 0):.2f}")
    except Exception as e:
        print(f"BG-NBD Prediction Failed: {e}")
    
    # Test batch prediction
    print("\nTesting batch prediction...")
    batch_customers = [
        sample_customer,
        {
            "customer_id": "TEST_CUSTOMER_002",
            "recency": 15,
            "frequency": 8,
            "monetary_value": 2400.0,
            "tenure": 365,
            "avg_order_value": 300.0,
            "days_between_purchases": 45.6,
            "total_orders": 8,
            "age": 42,
            "gender": "F",
            "country": "UK"
        }
    ]
    
    try:
        response = requests.post(f"{BASE_URL}/predict/batch?model_type=xgboost", json=batch_customers)
        batch_result = response.json()
        print(f"Batch Prediction: {response.status_code}")
        if response.status_code == 200:
            predictions = batch_result.get("predictions", [])
            print(f"Predicted {len(predictions)} customers")
            for pred in predictions:
                print(f"Customer {pred.get('customer_id')}: ${pred.get('predicted_clv', 0):.2f}")
    except Exception as e:
        print(f"Batch Prediction Failed: {e}")
    
    # Test model info
    print("\nTesting model info...")
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        model_info = response.json()
        print(f"Model Info: {response.status_code}")
        if response.status_code == 200:
            print(f"Model name: {model_info.get('model_name')}")
            print(f"Model type: {model_info.get('model_type')}")
            print(f"Features: {len(model_info.get('features', []))}")
    except Exception as e:
        print(f"Model Info Failed: {e}")
    
    # Test model performance
    try:
        response = requests.get(f"{BASE_URL}/model/performance")
        performance = response.json()
        print(f"Model Performance: {response.status_code}")
        if response.status_code == 200:
            print(f"Models trained: {performance.get('models_trained', {})}")
            print(f"Performance metrics: {performance.get('model_performance', {})}")
    except Exception as e:
        print(f"Model Performance Failed: {e}")
    
    print("\nAPI testing completed!")

if __name__ == "__main__":
    print("Make sure the CLV Predictor API is running on http://localhost:8002")
    print("You can start it with: python -m clv-predictor.app.main")
    print()
    test_api_endpoints()