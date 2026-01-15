import requests
import json
import time
from datetime import datetime, timedelta

def test_api():
    """Test the Product Demand Forecasting API"""
    base_url = "http://localhost:8003"
    
    print("🚀 Testing Product Demand Forecasting API")
    print("=" * 50)
    
    # Test 1: Health Check
    print("1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed: {health_data['status']}")
            print(f"   Available models: {health_data['available_models']}")
            print(f"   Loaded models: {health_data['loaded_models']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Test 2: Generate Sample Data
    print("\n2. Generating sample data...")
    try:
        response = requests.post(f"{base_url}/generate-sample-data?n_products=3&n_days=180")
        if response.status_code == 200:
            sample_data = response.json()
            print(f"✅ Generated {sample_data['total_products']} products with {sample_data['total_records']} records")
            
            # Get first product data
            first_product = list(sample_data['sample_products'].keys())[0]
            product_data = sample_data['sample_products'][first_product]
            print(f"   Sample product: {first_product} with {len(product_data)} data points")
            
            # Store for later tests
            test_product = first_product
            test_data = product_data
        else:
            print(f"❌ Sample data generation failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Sample data generation error: {e}")
        return
    
    # Test 3: Train ARIMA Model
    print(f"\n3. Training ARIMA model for {test_product}...")
    try:
        training_request = {
            "training_data": test_data[:120],  # Use first 120 days
            "model_type": "arima",
            "validation_split": 0.2
        }
        
        response = requests.post(f"{base_url}/train/arima", json=training_request)
        if response.status_code == 200:
            training_result = response.json()
            print(f"✅ ARIMA training completed: {training_result['message']}")
            if 'model_performance' in training_result and training_result['model_performance']:
                perf = training_result['model_performance']
                print(f"   MAPE: {perf.get('mape', 'N/A'):.2f}%")
                print(f"   RMSE: {perf.get('rmse', 'N/A'):.2f}")
        else:
            print(f"❌ ARIMA training failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ ARIMA training error: {e}")
    
    # Test 4: Train Prophet Model
    print(f"\n4. Training Prophet model for {test_product}...")
    try:
        training_request = {
            "training_data": test_data[:120],
            "model_type": "prophet",
            "validation_split": 0.2
        }
        
        response = requests.post(f"{base_url}/train/prophet", json=training_request)
        if response.status_code == 200:
            training_result = response.json()
            print(f"✅ Prophet training completed: {training_result['message']}")
        else:
            print(f"❌ Prophet training failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Prophet training error: {e}")
    
    # Test 5: Generate Forecast
    print(f"\n5. Generating forecast for {test_product}...")
    try:
        forecast_request = {
            "product_id": test_product,
            "historical_data": test_data[120:150],  # Use recent 30 days
            "forecast_periods": 14,
            "model_type": "prophet",
            "include_confidence_interval": True,
            "confidence_level": 0.95
        }
        
        response = requests.post(f"{base_url}/forecast", json=forecast_request)
        if response.status_code == 200:
            forecast_result = response.json()
            print(f"✅ Forecast generated successfully!")
            print(f"   Model: {forecast_result['model_type']}")
            print(f"   Periods: {forecast_result['forecast_periods']}")
            print(f"   First prediction: {forecast_result['forecast'][0]['demand']:.2f}")
            
            if forecast_result.get('confidence_intervals'):
                print(f"   Confidence intervals: Available")
        else:
            print(f"❌ Forecast failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Forecast error: {e}")
    
    # Test 6: Batch Forecast
    print(f"\n6. Testing batch forecast...")
    try:
        # Get second product for batch test
        if len(sample_data['sample_products']) >= 2:
            second_product = list(sample_data['sample_products'].keys())[1]
            second_data = sample_data['sample_products'][second_product]
            
            batch_request = {
                "products": [
                    {
                        "product_id": test_product,
                        "historical_data": test_data[120:150],
                        "forecast_periods": 7,
                        "model_type": "prophet"
                    },
                    {
                        "product_id": second_product,
                        "historical_data": second_data[120:150],
                        "forecast_periods": 7,
                        "model_type": "arima"
                    }
                ]
            }
            
            response = requests.post(f"{base_url}/forecast/batch", json=batch_request)
            if response.status_code == 200:
                batch_result = response.json()
                print(f"✅ Batch forecast completed!")
                print(f"   Products processed: {batch_result['total_products']}")
                print(f"   Processing time: {batch_result['processing_time']:.2f} seconds")
            else:
                print(f"❌ Batch forecast failed: {response.status_code}")
        else:
            print("⚠️  Skipping batch test - insufficient products")
    except Exception as e:
        print(f"❌ Batch forecast error: {e}")
    
    # Test 7: Model Info
    print(f"\n7. Getting model info for {test_product}...")
    try:
        response = requests.get(f"{base_url}/model/info/{test_product}")
        if response.status_code == 200:
            model_info = response.json()
            print(f"✅ Model info retrieved")
            print(f"   Model name: {model_info['model_name']}")
            print(f"   Model type: {model_info['model_type']}")
            print(f"   Features: {', '.join(model_info['features'][:3])}...")
        else:
            print(f"❌ Model info failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Model info error: {e}")
    
    # Test 8: Model Performance
    print(f"\n8. Getting overall model performance...")
    try:
        response = requests.get(f"{base_url}/model/performance")
        if response.status_code == 200:
            perf_info = response.json()
            print(f"✅ Model performance retrieved")
            print(f"   ARIMA models: {perf_info['models_available']['arima']}")
            print(f"   Prophet models: {perf_info['models_available']['prophet']}")
            print(f"   Total performance records: {len(perf_info['model_performance'])}")
        else:
            print(f"❌ Model performance failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Model performance error: {e}")
    
    # Test 9: Demand Pattern Analysis
    print(f"\n9. Analyzing demand pattern for {test_product}...")
    try:
        response = requests.post(f"{base_url}/analyze-demand-pattern", json=test_data[:90])
        if response.status_code == 200:
            pattern = response.json()
            print(f"✅ Demand pattern analyzed")
            print(f"   Trend: {pattern['trend']}")
            print(f"   Seasonality: {pattern['seasonality']}")
            print(f"   Volatility: {pattern['volatility']}")
        else:
            print(f"❌ Pattern analysis failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Pattern analysis error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 API testing completed!")
    print(f"📊 Tested endpoints: 9")
    print(f"🌐 Base URL: {base_url}")
    print(f"⏱️  Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    start_time = time.time()
    test_api()