# Customer Lifetime Value (CLV) Predictor API

A FastAPI-based service for predicting customer lifetime value using machine learning models.

## Features

- **XGBoost Regression**: Advanced gradient boosting for CLV prediction
- **BG-NBD Model**: Beta-Geometric/Negative Binomial Distribution for customer behavior modeling
- **Gamma-Gamma Model**: For monetary value estimation
- **Batch Prediction**: Predict CLV for multiple customers simultaneously
- **Synthetic Data Generation**: Generate sample customer data for testing and training
- **Model Performance Metrics**: Comprehensive evaluation of model performance

## API Endpoints

### Health and Information
- `GET /` - API information and available endpoints
- `GET /health` - Health check with model status
- `GET /model/info` - Detailed model information
- `GET /model/performance` - Model performance metrics

### Predictions
- `POST /predict` - Predict CLV for a single customer
- `POST /predict/batch` - Predict CLV for multiple customers

### Training and Data
- `POST /train/{model_type}` - Train a model (xgboost or bg_nbd)
- `POST /generate-sample-data` - Generate synthetic customer data

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the API:
```bash
python -m clv-predictor.app.main
```

The API will be available at `http://localhost:8002`

## Usage Examples

### Predict CLV for a Customer
```bash
curl -X POST "http://localhost:8002/predict?model_type=xgboost" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST_001",
    "recency": 30,
    "frequency": 5,
    "monetary_value": 1500.0,
    "tenure": 180,
    "avg_order_value": 300.0,
    "days_between_purchases": 36.0,
    "total_orders": 5
  }'
```

### Train a Model
```bash
curl -X POST "http://localhost:8002/train/xgboost?n_samples=1000"
```

### Generate Sample Data
```bash
curl -X POST "http://localhost:8002/generate-sample-data?n_customers=100"
```

## Customer Data Schema

The API expects customer data with the following fields:

- `customer_id` (str): Unique customer identifier
- `recency` (int): Days since last purchase
- `frequency` (int): Number of purchases
- `monetary_value` (float): Total amount spent
- `tenure` (int): Days since first purchase
- `avg_order_value` (float): Average order value
- `days_between_purchases` (float): Average days between purchases
- `total_orders` (int): Total number of orders
- `age` (int, optional): Customer age
- `gender` (str, optional): Customer gender
- `country` (str, optional): Customer country

## Models

### XGBoost Regression
- Uses gradient boosting for CLV prediction
- Features: RFM metrics, purchase patterns, customer demographics
- Performance metrics: RMSE, MAE, R² score

### BG-NBD (Buy Till You Die)
- Probabilistic model for customer behavior
- Estimates: Purchase probability, churn probability
- Combined with Gamma-Gamma for monetary value

## Testing

Run the test script to verify API functionality:
```bash
python test_api.py
```

## Model Performance

After training, models are evaluated using:
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error  
- **R² Score**: Coefficient of determination
- **Training samples**: Number of customers used for training

## Dependencies

- FastAPI: Web framework
- Uvicorn: ASGI server
- Pydantic: Data validation
- NumPy: Numerical computing
- Pandas: Data manipulation
- Scikit-learn: Machine learning utilities
- XGBoost: Gradient boosting
- Lifetimes: Customer lifetime value models
- Joblib: Model serialization