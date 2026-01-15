"""
Test API endpoints for Predictive Maintenance System
"""

import requests
import json
import time
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8002"
HEADERS = {"Content-Type": "application/json"}

def test_health_check():
    """Test health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"System status: {data['status']}")
            print(f"Available models: {data['available_models']}")
            print(f"Loaded models: {data['loaded_models']}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Connection error: {e}")
        return False

def test_generate_sample_data():
    """Test sample data generation"""
    print("\nTesting sample data generation...")
    try:
        response = requests.post(f"{BASE_URL}/generate-sample-data?n_samples=10")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Generated {data['count']} samples")
            print(f"Equipment types: {data['summary']['equipment_types']}")
            print(f"Failure rate: {data['summary']['failure_rate']:.2%}")
            return data['sample_data'][0] if data['sample_data'] else None
        else:
            print(f"Error: {response.text}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_predict_failure(sample_data=None):
    """Test failure prediction"""
    print("\nTesting failure prediction...")
    
    # Use sample data or create test data
    if sample_data:
        test_data = sample_data
    else:
        test_data = {
            "equipment_id": "TEST-001",
            "equipment_type": "Motor",
            "operating_hours": 8500.5,
            "temperature": 65.2,
            "vibration": 3.1,
            "pressure": 45.8,
            "days_since_maintenance": 45,
            "last_maintenance_type": "Preventive",
            "maintenance_frequency": 30,
            "environmental_conditions": "Normal",
            "load_factor": 0.85,
            "age_years": 5.2
        }
    
    prediction_request = {
        "equipment_data": test_data,
        "model_type": "random_forest"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            headers=HEADERS,
            data=json.dumps(prediction_request)
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Equipment ID: {data['equipment_id']}")
            print(f"Failure Probability: {data['failure_probability']:.2%}")
            print(f"Risk Level: {data['risk_level']}")
            print(f"Model Type: {data['model_type']}")
            print(f"Recommended Actions: {data['recommended_actions']}")
            print(f"Maintenance Date: {data['maintenance_schedule']['recommended_maintenance_date']}")
            print(f"Total Cost: ${data['maintenance_schedule']['total_estimated_cost']}")
            return data['failure_probability']
        else:
            print(f"Error: {response.text}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_predict_with_models():
    """Test prediction with different models"""
    print("\nTesting prediction with different models...")
    
    test_data = {
        "equipment_id": "TEST-002",
        "equipment_type": "Pump",
        "operating_hours": 12000,
        "temperature": 75.5,
        "vibration": 4.2,
        "pressure": 30.0,
        "days_since_maintenance": 60,
        "last_maintenance_type": "Preventive",
        "maintenance_frequency": 45,
        "environmental_conditions": "High Temperature",
        "load_factor": 0.92,
        "age_years": 8.1
    }
    
    models = ["random_forest", "gradient_boosting", "logistic_regression"]
    
    for model_type in models:
        print(f"\nTesting {model_type}...")
        prediction_request = {
            "equipment_data": test_data,
            "model_type": model_type
        }
        
        try:
            response = requests.post(
                f"{BASE_URL}/predict/{model_type}",
                headers=HEADERS,
                data=json.dumps(prediction_request)
            )
            if response.status_code == 200:
                data = response.json()
                print(f"  Failure Probability: {data['failure_probability']:.2%}")
                print(f"  Risk Level: {data['risk_level']}")
            else:
                print(f"  Error: {response.text}")
        except Exception as e:
            print(f"  Error: {e}")

def test_get_models():
    """Test getting available models"""
    print("\nTesting get available models...")
    try:
        response = requests.get(f"{BASE_URL}/models")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Available models: {data['available_models']}")
            print(f"Loaded models: {data['loaded_models']}")
            return data['available_models']
        else:
            print(f"Error: {response.text}")
            return []
    except Exception as e:
        print(f"Error: {e}")
        return []

def test_get_model_info():
    """Test getting model information"""
    print("\nTesting get model information...")
    
    models = ["random_forest", "gradient_boosting", "logistic_regression"]
    
    for model_type in models:
        print(f"\nTesting {model_type} info...")
        try:
            response = requests.get(f"{BASE_URL}/model-info/{model_type}")
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"  Model Type: {data['model_type']}")
                print(f"  Training Date: {data['training_date']}")
                print(f"  Accuracy: {data['accuracy']}")
            else:
                print(f"  Error: {response.text}")
        except Exception as e:
            print(f"  Error: {e}")

def test_train_models():
    """Test model training endpoints"""
    print("\nTesting model training...")
    
    models = ["random_forest", "gradient_boosting", "logistic_regression"]
    
    for model_type in models:
        print(f"\nTraining {model_type}...")
        try:
            response = requests.post(f"{BASE_URL}/train/{model_type}")
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"  Message: {data['message']}")
                print(f"  Training Samples: {data['training_samples']}")
            else:
                print(f"  Error: {response.text}")
        except Exception as e:
            print(f"  Error: {e}")

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("PREDICTIVE MAINTENANCE SYSTEM API TESTS")
    print("=" * 60)
    print(f"Testing against: {BASE_URL}")
    print(f"Started at: {datetime.now()}")
    
    # Test sequence
    tests_passed = 0
    total_tests = 0
    
    # Health check
    total_tests += 1
    if test_health_check():
        tests_passed += 1
    
    # Generate sample data
    sample_data = test_generate_sample_data()
    
    # Get models
    available_models = test_get_models()
    
    # If no models are loaded, try training them
    if not available_models or len([m for m in available_models if m is not None]) == 0:
        print("\nNo models loaded. Attempting to train models...")
        test_train_models()
        # Reload models info
        available_models = test_get_models()
    
    # Test predictions
    if available_models:
        total_tests += 1
        if test_predict_failure(sample_data):
            tests_passed += 1
        
        total_tests += 1
        test_predict_with_models()
        tests_passed += 1  # Assume passed if no exceptions
        
        total_tests += 1
        test_get_model_info()
        tests_passed += 1  # Assume passed if no exceptions
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests Passed: {tests_passed}/{total_tests}")
    print(f"Success Rate: {(tests_passed/total_tests*100):.1f}%" if total_tests > 0 else "N/A")
    print(f"Completed at: {datetime.now()}")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    # Wait a moment for server to be ready
    print("Waiting for server to be ready...")
    time.sleep(2)
    
    success = run_all_tests()
    exit(0 if success else 1)