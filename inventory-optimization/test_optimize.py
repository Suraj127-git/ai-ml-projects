import requests
import json

# Test the /optimize endpoint
test_data = {
    "product_data": {
        "product_id": "test_product_1",
        "product_name": "Test Product 1",
        "demand_rate": 10,
        "unit_cost": 50,
        "holding_cost_rate": 0.25,
        "ordering_cost": 100,
        "lead_time_days": 5,
        "demand_std": 10,
        "service_level": 0.95,
        "current_stock": 50,
        "category": "Electronics"
    },
    "optimization_method": "eoq"
}

print("Testing /optimize endpoint...")
try:
    response = requests.post("http://localhost:8005/optimize", json=test_data)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print("Success!")
        print(f"Results: {json.dumps(result, indent=2)}")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")