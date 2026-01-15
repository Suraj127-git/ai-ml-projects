import requests
import json
import numpy as np
from datetime import datetime, timedelta
import time

def create_sample_network():
    """Create a sample supply chain network for testing"""
    
    # Sample nodes
    nodes = [
        {
            "node_id": "supplier_1",
            "name": "Primary Supplier",
            "node_type": "supplier",
            "location": {"lat": 40.7128, "lng": -74.0060},
            "capacity": 1000,
            "setup_cost": 50000
        },
        {
            "node_id": "dc_1",
            "name": "East Coast Distribution Center",
            "node_type": "distribution_center",
            "location": {"lat": 39.9526, "lng": -75.1652},
            "capacity": 800,
            "setup_cost": 100000
        },
        {
            "node_id": "dc_2",
            "name": "West Coast Distribution Center",
            "node_type": "distribution_center",
            "location": {"lat": 34.0522, "lng": -118.2437},
            "capacity": 700,
            "setup_cost": 120000
        },
        {
            "node_id": "retailer_1",
            "name": "NYC Retailer",
            "node_type": "retailer",
            "location": {"lat": 40.7128, "lng": -74.0060},
            "capacity": 300,
            "setup_cost": 20000
        },
        {
            "node_id": "retailer_2",
            "name": "LA Retailer",
            "node_type": "retailer",
            "location": {"lat": 34.0522, "lng": -118.2437},
            "capacity": 250,
            "setup_cost": 18000
        }
    ]
    
    # Sample edges
    edges = [
        {
            "from_node": "supplier_1",
            "to_node": "dc_1",
            "capacity": 500,
            "transportation_cost": 2.5,
            "lead_time": 3,
            "distance": 150,
            "carbon_factor": 0.1,
            "reliability": 0.95
        },
        {
            "from_node": "supplier_1",
            "to_node": "dc_2",
            "capacity": 400,
            "transportation_cost": 3.2,
            "lead_time": 5,
            "distance": 280,
            "carbon_factor": 0.12,
            "reliability": 0.92
        },
        {
            "from_node": "dc_1",
            "to_node": "retailer_1",
            "capacity": 200,
            "transportation_cost": 1.8,
            "lead_time": 1,
            "distance": 80,
            "carbon_factor": 0.08,
            "reliability": 0.98
        },
        {
            "from_node": "dc_2",
            "to_node": "retailer_2",
            "capacity": 180,
            "transportation_cost": 2.1,
            "lead_time": 2,
            "distance": 50,
            "carbon_factor": 0.09,
            "reliability": 0.96
        }
    ]
    
    # Sample demands
    demands = [
        {
            "node_id": "retailer_1",
            "product_id": "product_1",
            "quantity": 150,
            "mean_demand": 150,
            "std_demand": 20,
            "probability_exceeding": 0.1
        },
        {
            "node_id": "retailer_2",
            "product_id": "product_1",
            "quantity": 120,
            "mean_demand": 120,
            "std_demand": 15,
            "probability_exceeding": 0.08
        }
    ]
    
    return nodes, edges, demands

def test_network_optimization():
    """Test network flow optimization"""
    print("Testing network flow optimization...")
    
    base_url = "http://localhost:8005"
    
    # Get sample network
    response = requests.get(f"{base_url}/generate-sample-network")
    if response.status_code == 200:
        sample_data = response.json()
        nodes = sample_data["nodes"]
        edges = sample_data["edges"]
        demands = sample_data["demands"]
    else:
        print(f"Failed to get sample network: {response.status_code}")
        return
    
    # Test 1: Minimize cost optimization
    print("\n1. Testing cost minimization...")
    optimization_request = {
        "nodes": nodes,
        "edges": edges,
        "demands": demands,
        "objective": "minimize_cost",
        "constraints": {}
    }
    
    response = requests.post(f"{base_url}/optimize-network", json=optimization_request)
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Cost optimization successful")
        print(f"   - Total cost: ${result['total_cost']:.2f}")
        print(f"   - Execution time: {result['execution_time']:.2f}s")
        print(f"   - Status: {result['optimization_status']}")
        print(f"   - Optimal flows: {len(result['optimal_flows'])}")
    else:
        print(f"   ✗ Cost optimization failed: {response.status_code}")
        print(f"   - Error: {response.text}")
    
    # Test 2: Minimize time optimization
    print("\n2. Testing time minimization...")
    optimization_request["objective"] = "minimize_time"
    
    response = requests.post(f"{base_url}/optimize-network", json=optimization_request)
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Time optimization successful")
        print(f"   - Total cost: ${result['total_cost']:.2f}")
        print(f"   - Execution time: {result['execution_time']:.2f}s")
        print(f"   - Status: {result['optimization_status']}")
    else:
        print(f"   ✗ Time optimization failed: {response.status_code}")
    
    # Test 3: Minimize risk optimization
    print("\n3. Testing risk minimization...")
    optimization_request["objective"] = "minimize_risk"
    
    response = requests.post(f"{base_url}/optimize-network", json=optimization_request)
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Risk optimization successful")
        print(f"   - Total cost: ${result['total_cost']:.2f}")
        print(f"   - Execution time: {result['execution_time']:.2f}s")
        print(f"   - Status: {result['optimization_status']}")
    else:
        print(f"   ✗ Risk optimization failed: {response.status_code}")

def test_facility_location():
    """Test facility location optimization"""
    print("\n\nTesting facility location optimization...")
    
    base_url = "http://localhost:8005"
    
    # Get sample network
    response = requests.get(f"{base_url}/generate-sample-network")
    if response.status_code != 200:
        print(f"Failed to get sample network: {response.status_code}")
        return
    
    sample_data = response.json()
    nodes = sample_data["nodes"]
    demands = sample_data["demands"]
    
    # Test facility location optimization
    budget_constraint = 200000  # $200k budget
    
    response = requests.post(
        f"{base_url}/optimize-facility-location",
        json={
            "nodes": nodes,
            "demands": demands,
            "budget_constraint": budget_constraint
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Facility location optimization successful")
        print(f"   - Selected facilities: {result['selected_facilities']}")
        print(f"   - Total cost: ${result['total_cost']:.2f}")
        print(f"   - Status: {result['status']}")
        print(f"   - Assignments: {len(result['assignments'])}")
    else:
        print(f"   ✗ Facility location optimization failed: {response.status_code}")
        print(f"   - Error: {response.text}")

def test_inventory_optimization():
    """Test inventory optimization"""
    print("\n\nTesting inventory optimization...")
    
    base_url = "http://localhost:8005"
    
    # Create sample demand forecasts
    demand_forecasts = [
        {
            "node_id": "retailer_1",
            "product_id": "product_1",
            "quantity": 150,
            "mean_demand": 150,
            "std_demand": 20,
            "probability_exceeding": 0.1
        },
        {
            "node_id": "retailer_2",
            "product_id": "product_1",
            "quantity": 120,
            "mean_demand": 120,
            "std_demand": 15,
            "probability_exceeding": 0.08
        },
        {
            "node_id": "retailer_3",
            "product_id": "product_2",
            "quantity": 200,
            "mean_demand": 200,
            "std_demand": 30,
            "probability_exceeding": 0.12
        }
    ]
    
    holding_cost_rate = 0.2  # 20%
    shortage_cost_rate = 0.5  # 50%
    
    response = requests.post(
        f"{base_url}/optimize-inventory",
        json={
            "demand_forecasts": demand_forecasts,
            "holding_cost_rate": holding_cost_rate,
            "shortage_cost_rate": shortage_cost_rate
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Inventory optimization successful")
        print(f"   - Total holding cost: ${result['total_holding_cost']:.2f}")
        print(f"   - Total shortage cost: ${result['total_shortage_cost']:.2f}")
        print(f"   - Optimized items: {len(result['inventory_optimization'])}")
        
        for item in result['inventory_optimization']:
            print(f"   - {item['node_id']}: Optimal inventory = {item['optimal_inventory']:.1f}, Service level = {item['service_level']:.2%}")
    else:
        print(f"   ✗ Inventory optimization failed: {response.status_code}")

def test_demand_forecasting():
    """Test demand forecasting"""
    print("\n\nTesting demand forecasting...")
    
    base_url = "http://localhost:8005"
    
    # Test 1: Train demand forecast model
    print("\n1. Training demand forecast model...")
    response = requests.post(f"{base_url}/train-demand-forecast")
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Model training successful")
        print(f"   - Training samples: {result['training_samples']}")
        print(f"   - Training date: {result['training_date']}")
        
        if 'training_results' in result:
            for model_name, metrics in result['training_results'].items():
                print(f"   - {model_name}: RMSE = {metrics['rmse']:.2f}, MAE = {metrics['mae']:.2f}")
    else:
        print(f"   ✗ Model training failed: {response.status_code}")
    
    # Test 2: Predict demand
    print("\n2. Testing demand prediction...")
    product_features = {
        "price": 50.0,
        "seasonality": 0.5,
        "promotion": 1,
        "competitor_price": 45.0,
        "economic_index": 1.05
    }
    
    response = requests.post(
        f"{base_url}/predict-demand",
        json={
            "product_features": product_features,
            "model_type": "random_forest"
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Demand prediction successful")
        print(f"   - Predicted demand: {result['predicted_demand']:.1f} units")
        print(f"   - Model type: {result['model_type']}")
    else:
        print(f"   ✗ Demand prediction failed: {response.status_code}")

def test_health_check():
    """Test health check endpoint"""
    print("\n\nTesting health check...")
    
    base_url = "http://localhost:8005"
    
    response = requests.get(f"{base_url}/health")
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Health check successful")
        print(f"   - Status: {result['status']}")
        print(f"   - Pulp available: {result.get('pulp_available', False)}")
        print(f"   - Models loaded: {result.get('models_loaded', 0)}")
    else:
        print(f"   ✗ Health check failed: {response.status_code}")

def main():
    """Run all tests"""
    print("Supply Chain Optimization API Tests")
    print("=" * 40)
    
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(2)
    
    # Run tests
    test_health_check()
    test_network_optimization()
    test_facility_location()
    test_inventory_optimization()
    test_demand_forecasting()
    
    print("\n" + "=" * 40)
    print("All tests completed!")

if __name__ == "__main__":
    main()