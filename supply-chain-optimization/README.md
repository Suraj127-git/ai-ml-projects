# Supply Chain Optimization with Linear Programming

A comprehensive API for supply chain network optimization using linear programming and machine learning techniques. This system provides tools for network flow optimization, facility location planning, demand forecasting, and inventory optimization.

## Features

### 🔗 Network Flow Optimization
- **Linear Programming**: Uses PuLP for optimal network flow solutions
- **Multiple Objectives**: Minimize cost, time, risk, or balance cost-service
- **Constraint Handling**: Capacity constraints, demand satisfaction, flow conservation
- **Fallback Heuristics**: Greedy algorithms when advanced solvers unavailable

### 🏭 Facility Location Optimization
- **Mixed-Integer Programming**: Optimal facility placement
- **Budget Constraints**: Respect setup cost limitations
- **Demand Assignment**: Efficient customer-facility assignment
- **Capacity Planning**: Ensure facilities can handle assigned demand

### 📈 Demand Forecasting
- **Machine Learning Models**: Random Forest and Linear Regression
- **Feature Engineering**: Price, seasonality, promotions, competitor analysis
- **Model Evaluation**: MSE, MAE, RMSE metrics
- **Synthetic Data**: Automatic data generation for testing

### 📦 Inventory Optimization
- **Newsvendor Model**: Optimal inventory levels under uncertainty
- **Service Level Optimization**: Balance holding vs shortage costs
- **Risk Management**: Probability-based inventory decisions
- **Cost Analysis**: Holding and shortage cost calculations

## API Endpoints

### Network Optimization
- `POST /optimize-network` - Optimize network flow
- `POST /optimize-facility-location` - Optimize facility placement
- `GET /generate-sample-network` - Get sample network data

### Demand Management
- `POST /train-demand-forecast` - Train demand forecasting models
- `POST /predict-demand` - Predict future demand
- `POST /optimize-inventory` - Optimize inventory levels

### System
- `GET /health` - Health check
- `GET /model-info` - Get model information

## Quick Start

### 1. Installation

```bash
# Clone or download the project
cd supply-chain-optimization

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
# Start the FastAPI server
uvicorn app.main:app --reload --port 8005

# Or use the Python module
python -m app.main
```

### 3. Test the API

```bash
# Run comprehensive tests
python test_api.py

# Or validate the setup
python validate.py
```

## Usage Examples

### Network Flow Optimization

```python
import requests

# Define your supply chain network
nodes = [
    {
        "node_id": "supplier_1",
        "name": "Primary Supplier",
        "node_type": "supplier",
        "location": {"lat": 40.7128, "lng": -74.0060},
        "capacity": 1000,
        "setup_cost": 50000
    },
    # ... more nodes
]

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
    # ... more edges
]

demands = [
    {
        "node_id": "retailer_1",
        "product_id": "product_1",
        "quantity": 150,
        "mean_demand": 150,
        "std_demand": 20,
        "probability_exceeding": 0.1
    },
    # ... more demands
]

# Optimize network flow
response = requests.post("http://localhost:8005/optimize-network", json={
    "nodes": nodes,
    "edges": edges,
    "demands": demands,
    "objective": "minimize_cost"
})

result = response.json()
print(f"Optimal cost: ${result['total_cost']}")
print(f"Optimal flows: {result['optimal_flows']}")
```

### Demand Forecasting

```python
# Train demand forecasting model
response = requests.post("http://localhost:8005/train-demand-forecast")
print(f"Training result: {response.json()}")

# Predict demand
product_features = {
    "price": 50.0,
    "seasonality": 0.5,
    "promotion": 1,
    "competitor_price": 45.0,
    "economic_index": 1.05
}

response = requests.post("http://localhost:8005/predict-demand", json={
    "product_features": product_features,
    "model_type": "random_forest"
})

print(f"Predicted demand: {response.json()['predicted_demand']} units")
```

### Inventory Optimization

```python
# Optimize inventory levels
demand_forecasts = [
    {
        "node_id": "retailer_1",
        "product_id": "product_1",
        "quantity": 150,
        "mean_demand": 150,
        "std_demand": 20,
        "probability_exceeding": 0.1
    }
]

response = requests.post("http://localhost:8005/optimize-inventory", json={
    "demand_forecasts": demand_forecasts,
    "holding_cost_rate": 0.2,
    "shortage_cost_rate": 0.5
})

result = response.json()
for item in result['inventory_optimization']:
    print(f"Optimal inventory for {item['node_id']}: {item['optimal_inventory']:.1f} units")
```

## Optimization Objectives

### 1. Minimize Cost
- Minimizes total transportation and operational costs
- Best for cost-sensitive supply chains
- Considers distance, transportation mode, and fuel costs

### 2. Minimize Time
- Minimizes total lead time across the network
- Ideal for time-sensitive products
- Prioritizes faster transportation modes

### 3. Minimize Risk
- Minimizes supply chain disruption risk
- Considers supplier reliability and route stability
- Best for critical supply chains

### 4. Balance Cost-Service
- Balances cost minimization with service level
- Maintains acceptable delivery performance
- Good for balanced supply chain strategies

## Model Architecture

### Linear Programming Engine
- **PuLP Integration**: Advanced linear programming solver
- **Constraint Programming**: Handles complex business rules
- **Multi-objective Optimization**: Balances competing objectives
- **Scalable Algorithms**: Handles large-scale networks

### Machine Learning Models
- **Random Forest**: Non-linear demand patterns
- **Linear Regression**: Simple trend analysis
- **Feature Engineering**: Automated feature creation
- **Model Validation**: Cross-validation and performance metrics

### Supply Chain Components
- **Nodes**: Suppliers, distribution centers, retailers
- **Edges**: Transportation links with costs and constraints
- **Products**: Items flowing through the network
- **Demands**: Customer requirements and forecasts

## Data Requirements

### Network Data
- Node locations and capacities
- Transportation costs and lead times
- Distance and carbon footprint data
- Reliability metrics

### Demand Data
- Historical demand patterns
- Product features and pricing
- Seasonal and promotional data
- Economic indicators

### Cost Parameters
- Holding cost rates
- Shortage cost rates
- Setup costs for facilities
- Transportation cost structures

## Performance Metrics

### Network Optimization
- **Total Cost**: Overall supply chain cost
- **Service Level**: Demand satisfaction rate
- **Utilization**: Node and edge capacity usage
- **Execution Time**: Algorithm performance

### Demand Forecasting
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of determination

### Inventory Optimization
- **Service Level**: Stock availability rate
- **Holding Cost**: Inventory carrying cost
- **Shortage Cost**: Stockout penalty cost
- **Turnover Rate**: Inventory efficiency

## Configuration

### Environment Variables
```bash
# Optional: Configure solver settings
PULP_SOLVER=CBC  # or GLPK, CPLEX, Gurobi
MAX_EXECUTION_TIME=300  # seconds
ENABLE_CACHING=true
```

### Model Parameters
```python
# Optimization parameters
OPTIMIZATION_TOLERANCE = 1e-6
MAX_ITERATIONS = 1000
CONSTRAINT_RELAXATION = 0.01

# ML model parameters
RANDOM_FOREST_ESTIMATORS = 100
LINEAR_REGRESSION_NORMALIZE = True
CROSS_VALIDATION_FOLDS = 5
```

## Limitations

- **PuLP Dependency**: Advanced optimization requires PuLP library
- **Data Quality**: Results depend on input data accuracy
- **Computational Complexity**: Large networks may require significant processing time
- **Model Assumptions**: Linear programming assumes linear relationships

## Future Enhancements

- **Advanced Solvers**: Integration with commercial solvers (CPLEX, Gurobi)
- **Stochastic Optimization**: Handle demand and supply uncertainty
- **Multi-period Planning**: Dynamic optimization over time
- **Real-time Integration**: Live data feeds and continuous optimization
- **Visualization Tools**: Network diagrams and performance dashboards

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.