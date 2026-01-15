# Price Optimization Engine with Reinforcement Learning

An intelligent pricing system that uses reinforcement learning and machine learning algorithms to optimize product prices dynamically based on market conditions, competitor analysis, and demand patterns.

## Features

- **Reinforcement Learning**: Q-learning agent for dynamic price optimization
- **Machine Learning Models**: Random Forest and Gradient Boosting for price prediction
- **Market Analysis**: Competitor price monitoring and elasticity analysis
- **Multi-Strategy Approach**: Combines RL, ML models, and rule-based pricing
- **Batch Processing**: Optimize prices for multiple products simultaneously
- **Performance Analytics**: Comprehensive metrics and insights

## API Endpoints

### Core Endpoints

- `POST /optimize-price` - Optimize price for a single product
- `POST /optimize-batch` - Optimize prices for multiple products
- `POST /analyze-elasticity` - Analyze price elasticity for products
- `GET /competitor-analysis/{product_id}` - Analyze competitor pricing
- `POST /demand-forecast` - Forecast demand for products
- `GET /price-history/{product_id}` - Get historical price data
- `GET /performance-metrics` - Get optimization performance metrics
- `GET /recommendations/{product_id}` - Get pricing recommendations
- `GET /health` - Health check endpoint

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Start the API Server

```bash
python -m app.main
```

The server will start on `http://localhost:8002`

### Example Usage

```python
import requests

# Single product optimization
response = requests.post("http://localhost:8002/optimize-price", json={
    "product_id": "laptop_001",
    "product_name": "Gaming Laptop Pro",
    "category": "electronics",
    "current_price": 1299.99,
    "cost_price": 800.00,
    "competitor_prices": [1199.99, 1349.99, 1275.00, 1325.50],
    "demand_history": [45, 52, 38, 41, 48, 55, 42],
    "inventory_level": 25,
    "seasonality_factor": 1.2,
    "price_elasticity": -1.5,
    "target_margin": 0.25,
    "market_conditions": "high_demand"
})

print(response.json())
```

### Batch Optimization

```python
# Multiple products optimization
response = requests.post("http://localhost:8002/optimize-batch", json={
    "products": [
        {
            "product_id": "laptop_001",
            "product_name": "Gaming Laptop Pro",
            "category": "electronics",
            "current_price": 1299.99,
            "cost_price": 800.00,
            "competitor_prices": [1199.99, 1349.99, 1275.00, 1325.50],
            "demand_history": [45, 52, 38, 41, 48, 55, 42],
            "inventory_level": 25,
            "seasonality_factor": 1.2,
            "price_elasticity": -1.5,
            "target_margin": 0.25,
            "market_conditions": "high_demand"
        },
        {
            "product_id": "shirt_002",
            "product_name": "Designer Cotton Shirt",
            "category": "clothing",
            "current_price": 79.99,
            "cost_price": 35.00,
            "competitor_prices": [69.99, 89.99, 75.00, 82.50],
            "demand_history": [120, 135, 98, 110, 125, 140, 105],
            "inventory_level": 200,
            "seasonality_factor": 0.9,
            "price_elasticity": -2.0,
            "target_margin": 0.35,
            "market_conditions": "normal"
        }
    ]
})

print(response.json())
```

## Reinforcement Learning

The system uses a Q-learning agent that:

- **States**: Market conditions, inventory levels, competitor prices
- **Actions**: Price adjustments (increase, decrease, maintain)
- **Rewards**: Revenue optimization, margin targets, market share
- **Exploration**: Epsilon-greedy strategy with decay

## Machine Learning Models

### Random Forest Regressor
- Predicts optimal prices based on historical data
- Handles non-linear relationships
- Feature importance analysis

### Gradient Boosting Regressor
- Sequential learning for price optimization
- Captures complex patterns in pricing data
- Ensemble approach for robust predictions

## Price Elasticity Analysis

The system calculates price elasticity using:

```python
def calculate_price_elasticity(self, price_changes, demand_changes):
    """Calculate price elasticity of demand"""
    if len(price_changes) < 2:
        return -1.0  # Default elasticity
    
    # Calculate percentage changes
    price_pct_changes = [(p2 - p1) / p1 for p1, p2 in zip(price_changes[:-1], price_changes[1:])]
    demand_pct_changes = [(d2 - d1) / d1 for d1, d2 in zip(demand_changes[:-1], demand_changes[1:])]
    
    # Calculate elasticity (demand change / price change)
    elasticities = []
    for price_change, demand_change in zip(price_pct_changes, demand_pct_changes):
        if price_change != 0:
            elasticity = demand_change / price_change
            elasticities.append(elasticity)
    
    return np.mean(elasticities) if elasticities else -1.0
```

## Testing

Run the test suite:

```bash
python test_api.py
```

## Performance Metrics

The system tracks:

- **Revenue Impact**: Price optimization effect on total revenue
- **Margin Optimization**: Achievement of target profit margins
- **Market Share**: Competitive positioning effectiveness
- **Demand Forecasting Accuracy**: Prediction quality metrics

## Configuration

Key parameters in the model:

- `epsilon`: Exploration rate for Q-learning (default: 0.1)
- `learning_rate`: Q-learning learning rate (default: 0.1)
- `discount_factor`: Future reward discount (default: 0.9)
- `n_estimators`: Number of trees in Random Forest (default: 100)
- `max_depth`: Maximum tree depth (default: 10)

## Data Requirements

Input data should include:

- Product identification and metadata
- Current pricing information
- Cost structure data
- Competitor pricing intelligence
- Historical demand patterns
- Inventory levels
- Market condition indicators

## License

This project is part of the AI/ML Projects collection.