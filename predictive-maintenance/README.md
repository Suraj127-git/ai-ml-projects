# Predictive Maintenance System

A comprehensive machine learning system for predicting equipment failures and optimizing maintenance schedules using multiple algorithms including Random Forest, Gradient Boosting, and Logistic Regression.

## Features

- **Multi-Model Support**: Random Forest, Gradient Boosting, and Logistic Regression
- **Equipment Failure Prediction**: Predict probability of equipment failure
- **Maintenance Scheduling**: Generate optimized maintenance schedules
- **Risk Assessment**: Classify equipment into low, medium, and high risk categories
- **Batch Processing**: Support for multiple equipment predictions
- **Comprehensive API**: RESTful API with detailed documentation
- **Synthetic Data Generation**: Built-in data generation for testing and development

## API Endpoints

### Core Prediction Endpoints

#### 1. Predict Equipment Failure
```http
POST /predict
```
Predict equipment failure probability and generate maintenance recommendations.

**Request Body:**
```json
{
  "equipment_data": {
    "equipment_id": "PUMP-001",
    "equipment_type": "Pump",
    "operating_hours": 8500.5,
    "temperature": 65.2,
    "vibration": 3.1,
    "pressure": 45.8,
    "days_since_maintenance": 45,
    "last_maintenance_type": "Preventive",
    "maintenance_frequency": 30,
    "environmental_conditions": "Normal",
    "load_factor": 0.85,
    "age_years": 5.2
  },
  "model_type": "random_forest"
}
```

**Response:**
```json
{
  "equipment_id": "PUMP-001",
  "failure_probability": 0.73,
  "risk_level": "high",
  "maintenance_schedule": {
    "recommended_maintenance_date": "2024-01-15",
    "urgency_level": "immediate",
    "maintenance_tasks": [
      {
        "task_type": "Inspection",
        "description": "Check pump seals and bearings",
        "priority": "High",
        "estimated_duration_hours": 2.0,
        "estimated_cost": 150.0
      }
    ],
    "total_estimated_cost": 150.0,
    "total_estimated_duration": 2.0
  },
  "recommended_actions": [
    "Schedule immediate maintenance",
    "Monitor temperature and vibration closely",
    "Consider equipment replacement if maintenance cost exceeds threshold"
  ],
  "model_type": "random_forest",
  "prediction_timestamp": "2024-01-01T10:30:00"
}
```

#### 2. Predict with Specific Model
```http
POST /predict/{model_type}
```
Predict using a specific model (random_forest, gradient_boosting, logistic_regression).

### Model Management

#### 3. Get Available Models
```http
GET /models
```
Get list of available prediction models.

#### 4. Get Model Information
```http
GET /model-info/{model_type}
```
Get detailed information about a specific model.

### Data Generation

#### 5. Generate Sample Data
```http
POST /generate-sample-data?n_samples=100
```
Generate synthetic maintenance data for testing.

### System Health

#### 6. Health Check
```http
GET /health
```
Check system health and loaded models.

### Training (Development)

#### 7. Train Model
```http
POST /train/{model_type}
```
Train a specific model (for development purposes).

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd predictive-maintenance
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the API Server

```bash
python -m app.main
```

The API will be available at `http://localhost:8002`

### Training Models (Development)

To train models with synthetic data:

```bash
# Train Random Forest model
curl -X POST http://localhost:8002/train/random_forest

# Train Gradient Boosting model  
curl -X POST http://localhost:8002/train/gradient_boosting

# Train Logistic Regression model
curl -X POST http://localhost:8002/train/logistic_regression
```

### Making Predictions

```bash
# Predict using default model (Random Forest)
curl -X POST http://localhost:8002/predict \
  -H "Content-Type: application/json" \
  -d '{
    "equipment_data": {
      "equipment_id": "MOTOR-002",
      "equipment_type": "Motor",
      "operating_hours": 12000,
      "temperature": 75.5,
      "vibration": 4.2,
      "pressure": 30.0,
      "days_since_maintenance": 60,
      "last_maintenance_type": "Preventive",
      "maintenance_frequency": 45,
      "environmental_conditions": "High Temperature",
      "load_factor": 0.92,
      "age_years": 8.1
    }
  }'

# Predict using specific model
curl -X POST http://localhost:8002/predict/gradient_boosting \
  -H "Content-Type: application/json" \
  -d '{
    "equipment_data": {
      "equipment_id": "COMP-003",
      "equipment_type": "Compressor",
      "operating_hours": 9500,
      "temperature": 82.3,
      "vibration": 5.1,
      "pressure": 85.7,
      "days_since_maintenance": 35,
      "last_maintenance_type": "Condition-Based",
      "maintenance_frequency": 30,
      "environmental_conditions": "Dusty",
      "load_factor": 0.78,
      "age_years": 6.5
    }
  }'
```

## Equipment Types Supported

- **Pump**: Fluid handling equipment
- **Motor**: Electric motors and drives
- **Compressor**: Air and gas compression systems
- **Fan**: Ventilation and cooling fans
- **Bearing**: Rotating equipment bearings
- **Gearbox**: Mechanical power transmission

## Risk Levels

- **Low Risk** (0-40%): Equipment is operating normally
- **Medium Risk** (40-70%): Monitor equipment closely, plan maintenance
- **High Risk** (70-100%): Immediate maintenance required

## Maintenance Types

- **Preventive**: Scheduled maintenance to prevent failures
- **Corrective**: Repair after failure detection
- **Predictive**: Maintenance based on condition monitoring
- **Condition-Based**: Maintenance triggered by specific conditions

## Environmental Conditions

- **Normal**: Standard operating environment
- **Harsh**: Extreme temperatures, vibration, or contamination
- **Corrosive**: Chemical exposure or corrosive atmosphere
- **High Temperature**: Elevated ambient temperatures
- **High Humidity**: Moisture-rich environment
- **Dusty**: Particulate contamination

## Model Performance

The system uses multiple machine learning algorithms:

- **Random Forest**: Ensemble method with high accuracy
- **Gradient Boosting**: Sequential learning with excellent performance
- **Logistic Regression**: Simple, interpretable baseline model

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8002/docs`
- ReDoc: `http://localhost:8002/redoc`

## Development

### Project Structure
```
predictive-maintenance/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application
│   ├── model.py         # Core ML models
│   └── schemas.py       # Pydantic schemas
├── requirements.txt     # Dependencies
└── README.md           # Documentation
```

### Adding New Models

1. Implement model training in `app/model.py`
2. Add model loading in `app/main.py`
3. Update schemas if needed in `app/schemas.py`
4. Add model training endpoint

### Testing

Generate sample data and test predictions:

```bash
# Generate test data
curl -X POST "http://localhost:8002/generate-sample-data?n_samples=50"

# Test health endpoint
curl http://localhost:8002/health
```

## License

This project is part of the AI/ML Projects collection.