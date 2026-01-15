# Inventory Optimization System

An advanced inventory management system that optimizes stock levels using Economic Order Quantity (EOQ), Reorder Point (ROP), ABC analysis, and multi-echelon optimization techniques.

## Features

- **Economic Order Quantity (EOQ)**: Calculate optimal order quantities to minimize total inventory costs
- **Reorder Point (ROP)**: Determine when to reorder based on demand and lead time
- **Safety Stock Calculation**: Compute safety stock levels for desired service levels
- **ABC Analysis**: Classify products based on value and usage frequency
- **Multi-Echelon Optimization**: Optimize inventory across multiple locations
- **Stock Alerts**: Generate alerts for low stock and expiration warnings
- **Demand Forecasting**: Predict future demand using statistical methods
- **Performance Metrics**: Track inventory turnover, carrying costs, and service levels

## API Endpoints

### Core Optimization
- `POST /optimize` - Optimize inventory for single product
- `POST /optimize/batch` - Optimize inventory for multiple products
- `POST /abc-analysis` - Perform ABC analysis on product portfolio
- `POST /multi-echelon` - Optimize inventory across multiple locations

### Analysis & Monitoring
- `GET /stock-alerts` - Get stock level alerts
- `POST /demand-forecast` - Forecast demand for products
- `POST /calculate-metrics` - Calculate inventory performance metrics

### Utilities
- `GET /` - API information
- `GET /health` - Health check
- `POST /generate-sample-data` - Generate sample inventory data
- `GET /model/info` - Get model information

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Start the API Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8003
```

### Example Usage

#### Single Product Optimization
```python
import requests

# Optimize inventory for a single product
response = requests.post("http://localhost:8003/optimize", json={
    "product_id": "PROD001",
    "product_name": "Widget A",
    "category": "Electronics",
    "unit_cost": 25.0,
    "holding_cost_rate": 0.25,
    "ordering_cost": 100.0,
    "lead_time_days": 7,
    "current_stock": 150,
    "demand_rate": 500,
    "demand_std": 50,
    "service_level": 0.95,
    "expiration_days": 365,
    "min_order_quantity": 10,
    "max_stock_capacity": 1000,
    "supplier_reliability": 0.95
})

result = response.json()
print(f"Optimal Order Quantity: {result['economic_order_quantity']}")
print(f"Reorder Point: {result['reorder_point']}")
print(f"Safety Stock: {result['safety_stock']}")
```

#### Batch Optimization
```python
# Optimize inventory for multiple products
products = [
    {
        "product_id": "PROD001",
        "product_name": "Widget A",
        "category": "Electronics",
        "unit_cost": 25.0,
        "holding_cost_rate": 0.25,
        "ordering_cost": 100.0,
        "lead_time_days": 7,
        "current_stock": 150,
        "demand_rate": 500,
        "demand_std": 50,
        "service_level": 0.95
    },
    {
        "product_id": "PROD002",
        "product_name": "Gadget B",
        "category": "Accessories",
        "unit_cost": 15.0,
        "holding_cost_rate": 0.20,
        "ordering_cost": 75.0,
        "lead_time_days": 5,
        "current_stock": 200,
        "demand_rate": 300,
        "demand_std": 30,
        "service_level": 0.90
    }
]

response = requests.post("http://localhost:8003/optimize/batch", json={"products": products})
results = response.json()
```

#### ABC Analysis
```python
# Perform ABC analysis on product portfolio
response = requests.post("http://localhost:8003/abc-analysis", json={
    "products": products,
    "analysis_period_days": 365
})

analysis = response.json()
for category, items in analysis['abc_categories'].items():
    print(f"Category {category}: {len(items)} products")
```

## Model Details

### EOQ (Economic Order Quantity)
The EOQ model determines the optimal order quantity that minimizes the total inventory costs:
- Ordering costs
- Holding costs
- Shortage costs

### ROP (Reorder Point)
The ROP model calculates when to place new orders based on:
- Average demand during lead time
- Safety stock requirements
- Service level targets

### ABC Analysis
Classifies products into three categories:
- **A Items**: High value, low frequency (20% of items, 80% of value)
- **B Items**: Medium value, medium frequency (30% of items, 15% of value)
- **C Items**: Low value, high frequency (50% of items, 5% of value)

### Multi-Echelon Optimization
Optimizes inventory across multiple locations considering:
- Inter-location demand patterns
- Transportation costs
- Lead time variations
- Stock balancing opportunities

## Performance Metrics

The system tracks key inventory performance indicators:
- Inventory turnover ratio
- Carrying cost percentage
- Stockout frequency
- Service level achievement
- Order fulfillment rate
- Stock aging analysis

## Testing

Run the API tests:
```bash
python test_api.py
```

## Model Training

For advanced demand forecasting models:
```bash
cd notebooks
python train_inventory_models.py
```

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8003/docs`
- ReDoc: `http://localhost:8003/redoc`

## License

This project is part of the AI/ML Projects collection.