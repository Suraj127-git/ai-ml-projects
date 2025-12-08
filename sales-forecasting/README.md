# Sales Forecasting with Prophet

A comprehensive sales forecasting system using Prophet, ARIMA, and Linear Regression models to predict future sales trends with high accuracy.

## Features

- **Multiple Forecasting Models**: Prophet, ARIMA, and Linear Regression
- **Real-time Predictions**: FastAPI-based REST API for instant forecasts
- **Synthetic Data Generation**: Built-in data generation for testing and demonstration
- **Model Performance Metrics**: Comprehensive evaluation with RMSE, MAE, and MAPE
- **Batch Forecasting**: Support for bulk predictions
- **Interactive Documentation**: Auto-generated API documentation with Swagger UI

## Models Used

### Prophet
- Facebook's open-source forecasting tool
- Handles seasonality, trends, and holidays automatically
- Excellent for business time series with multiple seasonal patterns

### ARIMA
- Autoregressive Integrated Moving Average
- Classical statistical approach for time series forecasting
- Good for stationary time series data

### Linear Regression
- Simple baseline model with polynomial features
- Fast training and prediction
- Useful for trend-based forecasting

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Start the API Server
```bash
python -m app.main
```

### API Endpoints

- **POST /predict**: Single sales forecast
- **POST /predict/batch**: Batch sales forecasts
- **POST /train**: Train forecasting models
- **GET /model/info**: Get model information
- **GET /health**: Health check endpoint
- **POST /generate-sample-data**: Generate synthetic sales data

### Example API Call
```bash
curl -X POST "http://localhost:8001/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2024-01-15",
    "model_type": "prophet",
    "days_ahead": 30
  }'
```

## Training Models

The system supports training multiple models:

```python
# Train Prophet model
POST /train with {"model_type": "prophet"}

# Train ARIMA model  
POST /train with {"model_type": "arima"}

# Train Linear Regression model
POST /train with {"model_type": "linear_regression"}
```

## Model Performance

Each model provides performance metrics:
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error  
- **MAPE**: Mean Absolute Percentage Error

## Data Requirements

Input data should include:
- Date column (daily frequency)
- Sales values (numeric)
- Optional promotional events
- Optional external factors (weather, holidays, etc.)

## Synthetic Data Generation

The system can generate realistic synthetic sales data with:
- Trend components
- Weekly seasonality
- Monthly seasonality
- Yearly seasonality
- Promotional spikes
- Random noise

## File Structure

```
sales-forecasting/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application
│   ├── model.py         # Forecasting models
│   └── schemas.py       # Pydantic schemas
├── notebooks/
│   └── train_sales_forecast.ipynb  # Training notebook
├── requirements.txt     # Dependencies
├── README.md           # Documentation
└── test_api.py         # API tests
```

## Dependencies

- FastAPI: Web framework
- Prophet: Facebook's forecasting tool
- Statsmodels: ARIMA implementation
- Scikit-learn: Linear regression
- Pandas: Data manipulation
- NumPy: Numerical computing
- Joblib: Model serialization

## License

MIT License