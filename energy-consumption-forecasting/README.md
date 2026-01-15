# Energy Consumption Forecasting System

A comprehensive energy consumption forecasting system that uses multiple machine learning models (LSTM, Prophet, XGBoost) to predict energy usage patterns for different building types.

## Features

- **Multiple Forecasting Models**: LSTM, Prophet, and XGBoost models for accurate energy consumption predictions
- **Building Type Support**: Optimized models for office, residential, retail, industrial, hospital, and school buildings
- **Multi-Hour Forecasting**: Predict energy consumption from 1 hour to 168 hours (1 week) ahead
- **Confidence Intervals**: Statistical confidence bounds for forecast reliability
- **Anomaly Detection**: Identify unusual energy consumption patterns
- **Efficiency Analysis**: Comprehensive energy efficiency metrics and scoring
- **Batch Forecasting**: Compare multiple models simultaneously
- **Real-time Data Processing**: Handle streaming energy consumption data

## API Endpoints

### Core Forecasting
- `POST /forecast` - Generate energy consumption forecast
- `POST /forecast/batch` - Batch forecast with multiple models
- `POST /compare-models` - Compare performance of different models

### Analysis & Monitoring
- `POST /detect-anomalies` - Detect anomalies in energy consumption
- `POST /efficiency-analysis` - Calculate energy efficiency metrics
- `GET /models` - Get available forecasting models
- `GET /health` - Health check endpoint

### Data Management
- `POST /generate-sample-data` - Generate synthetic energy data for testing
- `POST /train/{model_type}` - Train specific forecasting model

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd energy-consumption-forecasting

# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

## Quick Start

### 1. Generate Sample Data
```python
import requests

# Generate 30 days of sample data for an office building
response = requests.post(
    "http://localhost:8002/generate-sample-data",
    params={"days": 30, "building_type": "office"}
)
sample_data = response.json()
```

### 2. Create Historical Data for Forecasting
```python
historical_data = [
    {
        "timestamp": "2024-01-01T00:00:00",
        "energy_consumption_kwh": 45.2,
        "temperature_celsius": 22.5,
        "humidity_percent": 65,
        "occupancy_rate": 0.8,
        "weather_condition": "sunny",
        "is_holiday": False
    },
    # ... more data points
]
```

### 3. Generate Forecast
```python
forecast_request = {
    "historical_data": historical_data,
    "forecast_hours": 24,
    "model_type": "xgboost",
    "building_type": "office",
    "include_confidence_intervals": True
}

response = requests.post(
    "http://localhost:8002/forecast",
    json=forecast_request
)

forecast_result = response.json()
print(f"Total forecast consumption: {forecast_result['total_forecast_consumption']} kWh")
print(f"Peak consumption hour: {forecast_result['peak_consumption_hour']}")
```

## Model Types

### XGBoost Model
- **Best for**: Short-term forecasting (1-24 hours)
- **Features**: Time-based features, weather data, occupancy patterns
- **Accuracy**: High accuracy for hourly predictions
- **Speed**: Fast inference

### Prophet Model
- **Best for**: Medium-term forecasting (24-168 hours)
- **Features**: Seasonal patterns, trend analysis, external regressors
- **Accuracy**: Excellent for weekly patterns
- **Speed**: Moderate inference speed

### LSTM Model
- **Best for**: Complex temporal patterns
- **Features**: Sequential data processing, long-term dependencies
- **Accuracy**: High accuracy for complex patterns
- **Speed**: Slower inference

## Building Types

The system supports six building types with optimized consumption profiles:

1. **Office Buildings**: Peak consumption during business hours (8 AM - 5 PM)
2. **Residential**: Morning and evening peaks, lower daytime consumption
3. **Retail**: Extended hours with higher weekend consumption
4. **Industrial**: Consistent high consumption with shift-based patterns
5. **Hospitals**: 24/7 operation with moderate peaks during day hours
6. **Schools**: Concentrated consumption during school hours

## Data Schema

### Energy Data Point
```json
{
    "timestamp": "2024-01-01T00:00:00",
    "energy_consumption_kwh": 45.2,
    "temperature_celsius": 22.5,
    "humidity_percent": 65,
    "occupancy_rate": 0.8,
    "weather_condition": "sunny",
    "is_holiday": false
}
```

### Forecast Request
```json
{
    "historical_data": [...],
    "forecast_hours": 24,
    "model_type": "xgboost",
    "building_type": "office",
    "include_confidence_intervals": true
}
```

### Forecast Response
```json
{
    "forecast": [...],
    "confidence_intervals": {
        "lower": [...],
        "upper": [...]
    },
    "model_performance": {
        "rmse": 2.5,
        "mae": 1.8,
        "mape": 5.2
    },
    "total_forecast_consumption": 1200.5,
    "average_hourly_consumption": 50.0,
    "peak_consumption_hour": "2024-01-02T14:00:00",
    "model_type": "xgboost",
    "forecast_timestamp": "2024-01-01T12:00:00"
}
```

## Performance Metrics

The system provides comprehensive performance metrics:

- **RMSE** (Root Mean Square Error): Overall prediction accuracy
- **MAE** (Mean Absolute Error): Average prediction error
- **MAPE** (Mean Absolute Percentage Error): Error as percentage
- **Load Factor**: Ratio of average to peak consumption
- **Efficiency Score**: Building-specific efficiency rating (0-10)

## Testing

Run the comprehensive test suite:

```bash
python test_api.py
```

The test suite includes:
- Health check validation
- Sample data generation
- Single model forecasting
- Batch forecasting with multiple models
- Model performance comparison
- Anomaly detection
- Efficiency analysis
- Error handling validation

## Configuration

### Environment Variables
- `MODEL_PATH`: Directory for storing trained models
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `MAX_FORECAST_HOURS`: Maximum allowed forecast hours (default: 168)

### Model Parameters
- `XGBOOST_N_ESTIMATORS`: Number of trees (default: 100)
- `XGBOOST_MAX_DEPTH`: Maximum tree depth (default: 6)
- `LSTM_SEQUENCE_LENGTH`: Input sequence length (default: 24)
- `PROPHET_INTERVAL_WIDTH`: Confidence interval width (default: 0.95)

## Deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8002

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
```

### Production Considerations
- Use a production ASGI server (Gunicorn with Uvicorn workers)
- Implement proper logging and monitoring
- Set up model retraining pipelines
- Configure rate limiting and authentication
- Use a reverse proxy (Nginx) for SSL termination

## API Examples

### Batch Forecast with Multiple Models
```python
batch_request = {
    "historical_data": historical_data,
    "forecast_hours": 48,
    "model_types": ["xgboost", "prophet"],
    "building_type": "office"
}

response = requests.post(f"{base_url}/forecast/batch", json=batch_request)
```

### Anomaly Detection
```python
anomaly_request = {
    "historical_data": historical_data,
    "model_type": "xgboost",
    "sensitivity": 0.95  # 95% confidence threshold
}

response = requests.post(f"{base_url}/detect-anomalies", json=anomaly_request)
```

### Energy Efficiency Analysis
```python
efficiency_request = {
    "historical_data": historical_data,
    "building_type": "office",
    "building_size_sqft": 50000,
    "occupancy_hours_per_day": 10
}

response = requests.post(f"{base_url}/efficiency-analysis", json=efficiency_request)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the GitHub repository
- Check the documentation and examples
- Review the test cases for usage patterns