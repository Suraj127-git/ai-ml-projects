# Product Demand Forecasting System

A comprehensive API for forecasting product demand using multiple time series models including ARIMA, Prophet, LSTM, and XGBoost.

## Features

- **Multiple Forecasting Models**: ARIMA, Prophet, LSTM, XGBoost
- **Batch Forecasting**: Process multiple products simultaneously
- **Confidence Intervals**: Generate prediction intervals for forecasts
- **Demand Pattern Analysis**: Identify trends, seasonality, and volatility
- **Model Performance Metrics**: MAE, RMSE, MAPE evaluation
- **Synthetic Data Generation**: Generate sample data for testing
- **RESTful API**: Easy integration with existing systems

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

1. **Start the API server:**
```bash
python -m app.main
```

2. **Generate sample data:**
```bash
curl -X POST "http://localhost:8003/generate-sample-data?n_products=3&n_days=365"
```

3. **Train a model:**
```bash
curl -X POST "http://localhost:8003/train/prophet" \
  -H "Content-Type: application/json" \
  -d @sample_training_data.json
```

4. **Generate forecast:**
```bash
curl -X POST "http://localhost:8003/forecast" \
  -H "Content-Type: application/json" \
  -d @forecast_request.json
```

## API Endpoints

### Health & Information
- `GET /` - API information
- `GET /health` - Health check and system status
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation (ReDoc)

### Forecasting
- `POST /forecast` - Generate demand forecast for a single product
- `POST /forecast/batch` - Generate forecasts for multiple products
- `POST /analyze-demand-pattern` - Analyze demand patterns in historical data

### Model Training
- `POST /train/{model_type}` - Train a forecasting model (arima/prophet/lstm/xgboost)
- `GET /model/info/{product_id}` - Get model information for a product
- `GET /model/performance` - Get overall model performance metrics

### Data Generation
- `POST /generate-sample-data` - Generate synthetic demand data for testing

## Model Types

### ARIMA (AutoRegressive Integrated Moving Average)
- Best for: Stable demand patterns with clear trends
- Features: Automatic stationarity testing, confidence intervals
- Parameters: (p=1, d=1, q=1) with automatic optimization

### Prophet
- Best for: Data with strong seasonality and holiday effects
- Features: Additive/multiplicative seasonality, holiday detection
- Regressors: Price, promotion, holiday effects

### LSTM (Long Short-Term Memory)
- Best for: Complex temporal patterns and long-term dependencies
- Features: Deep learning with sequence modeling
- Input: 30-day historical window with multiple features

### XGBoost
- Best for: Non-linear relationships and feature interactions
- Features: Gradient boosting with regularization
- Features: Price, promotion, seasonality, lag features

## Data Format

### Historical Data
```json
{
  "product_id": "PRODUCT_001",
  "date": "2023-01-01",
  "demand": 150.5,
  "price": 25.99,
  "promotion": 1,
  "seasonality": 10.2,
  "holiday": 0,
  "stock_level": 500
}
```

### Forecast Request
```json
{
  "product_id": "PRODUCT_001",
  "historical_data": [...],
  "forecast_periods": 30,
  "model_type": "prophet",
  "include_confidence_interval": true,
  "confidence_level": 0.95
}
```

### Training Request
```json
{
  "training_data": [...],
  "model_type": "prophet",
  "validation_split": 0.2,
  "hyperparameter_tuning": false
}
```

## Performance Metrics

- **MAE (Mean Absolute Error)**: Average absolute prediction error
- **RMSE (Root Mean Square Error)**: Square root of average squared errors
- **MAPE (Mean Absolute Percentage Error)**: Average percentage error
- **Confidence Intervals**: Prediction uncertainty bounds

## Configuration

### Environment Variables
- `PORT`: API server port (default: 8003)
- `HOST`: API server host (default: 0.0.0.0)
- `LOG_LEVEL`: Logging level (default: INFO)

### Model Parameters
Models are pre-configured with optimal hyperparameters but can be customized:
- ARIMA: Order parameters (p, d, q)
- Prophet: Seasonality modes, changepoint parameters
- LSTM: Layer sizes, dropout rates, learning rate
- XGBoost: Tree depth, learning rate, regularization

## Usage Examples

### Python Client
```python
import requests
import json

# Generate sample data
response = requests.post("http://localhost:8003/generate-sample-data?n_products=2&n_days=180")
sample_data = response.json()

# Train Prophet model
training_data = sample_data['sample_products']['PRODUCT_001']
training_request = {
    "training_data": training_data[:120],  # Use first 120 days for training
    "model_type": "prophet",
    "validation_split": 0.2
}

response = requests.post("http://localhost:8003/train/prophet", json=training_request)
print(f"Training result: {response.json()}")

# Generate forecast
forecast_request = {
    "product_id": "PRODUCT_001",
    "historical_data": training_data[120:],  # Use remaining data for forecast
    "forecast_periods": 30,
    "model_type": "prophet",
    "include_confidence_interval": True
}

response = requests.post("http://localhost:8003/forecast", json=forecast_request)
forecast = response.json()
print(f"Forecast: {forecast['forecast'][:3]}")  # Show first 3 predictions
```

### Demand Pattern Analysis
```python
# Analyze demand patterns
historical_data = sample_data['sample_products']['PRODUCT_001']

response = requests.post("http://localhost:8003/analyze-demand-pattern", json=historical_data)
pattern = response.json()
print(f"Demand pattern: {pattern['trend']} trend, {pattern['seasonality']} seasonality")
```

### Batch Forecasting
```python
# Batch forecast multiple products
batch_request = {
    "products": [
        {
            "product_id": "PRODUCT_001",
            "historical_data": sample_data['sample_products']['PRODUCT_001'][-30:],
            "forecast_periods": 14,
            "model_type": "prophet"
        },
        {
            "product_id": "PRODUCT_002", 
            "historical_data": sample_data['sample_products']['PRODUCT_002'][-30:],
            "forecast_periods": 14,
            "model_type": "arima"
        }
    ]
}

response = requests.post("http://localhost:8003/forecast/batch", json=batch_request)
batch_results = response.json()
print(f"Processed {batch_results['total_products']} products in {batch_results['processing_time']:.2f} seconds")
```

## Model Selection Guide

| Model | Best For | Advantages | Limitations |
|-------|----------|------------|-------------|
| ARIMA | Stable patterns, clear trends | Simple, interpretable | Assumes linear patterns |
| Prophet | Seasonal data, holidays | Handles seasonality well | Can overfit complex patterns |
| LSTM | Complex temporal patterns | Captures long-term dependencies | Requires more data, slower |
| XGBoost | Non-linear relationships | Fast, feature importance | Less interpretable |

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (invalid parameters)
- `404`: Resource not found
- `500`: Internal server error

Error responses include detailed messages:
```json
{
  "detail": "Model not trained for PRODUCT_001. Please train the model first."
}
```

## Dependencies

Core dependencies:
- `fastapi`: Web framework
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `scikit-learn`: Machine learning utilities
- `statsmodels`: Time series analysis (ARIMA)

Optional dependencies (install as needed):
- `prophet`: Facebook Prophet for time series forecasting
- `tensorflow`: Deep learning framework (for LSTM)
- `xgboost`: Gradient boosting framework

## Testing

Run the API tests:
```bash
python test_api.py
```

Test data generation and model training:
```bash
python notebooks/train_demand_models.py
```

## Deployment

### Local Development
```bash
python -m app.main
```

### Production
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8003 --workers 4
```

## License

This project is part of the ML Projects collection and follows the same licensing terms.

## Support

For issues and questions:
1. Check the API documentation at `/docs`
2. Review the error messages in API responses
3. Ensure all dependencies are properly installed
4. Verify data format matches the expected schema