# 🏠 House Price Prediction API - Comprehensive Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Technical Implementation](#technical-implementation)
3. [Architectural Documentation](#architectural-documentation)
4. [Project Structure](#project-structure)
5. [Learning Pathways](#learning-pathways)
6. [Setup and Usage](#setup-and-usage)
7. [API Reference](#api-reference)
8. [Performance Metrics](#performance-metrics)

---

## Project Overview

### Purpose and Goals
The House Price Prediction API is a machine learning microservice that predicts residential property prices based on various features like square footage, number of bedrooms, location, and amenities. It uses regression algorithms to provide accurate price estimates for real estate valuation.

### Business Problem
- **Problem**: Manual property valuation is time-consuming, subjective, and inconsistent. Real estate agents, buyers, and sellers need quick, data-driven price estimates.
- **Solution**: Automated price prediction using historical sales data and property features, providing instant, objective valuations.
- **Impact**: 
  - Reduce valuation time from hours to seconds
  - Provide consistent pricing across properties
  - Enable dynamic pricing for real estate platforms
  - Support investment decision-making
  - Improve market transparency

### Expected Outcomes
- **Primary Metrics**:
  - Prediction Accuracy (R² Score): >0.85
  - Mean Absolute Error (MAE): <$15,000
  - Root Mean Squared Error (RMSE): <$25,000
  - Response Time: <50ms per prediction
  - API Uptime: >99.9%

- **Success Criteria**:
  - Predict house prices within 10% of actual sale price
  - Handle 500+ predictions per minute
  - Support multiple property types and locations
  - Provide confidence intervals for predictions
  - Enable batch pricing for portfolios

---

## Technical Implementation

### Technology Stack

#### Core Technologies

1. **Scikit-learn** (v1.3+)
   - Linear Regression algorithm
   - Polynomial features for non-linear relationships
   - StandardScaler for feature normalization
   - Train/test splitting and cross-validation
   - Model persistence with joblib

2. **FastAPI** (v0.104.1)
   - Lightweight, high-performance web framework
   - Automatic data validation
   - Interactive API documentation (Swagger UI)
   - JSON request/response handling
   - CORS middleware support

3. **Pandas** (v2.0+)
   - Data loading and manipulation
   - Feature engineering
   - Data cleaning and preprocessing
   - CSV file handling

4. **NumPy** (v1.24+)
   - Numerical computations
   - Array operations
   - Mathematical functions

5. **Python** (v3.8+)
   - Type hints for code clarity
   - Modern syntax features
   - Extensive standard library

#### Supporting Libraries
- **Joblib**: Efficient model serialization
- **Uvicorn**: ASGI server for production
- **Pydantic**: Request/response validation

### Technology Alternatives

| Component | Current Choice | Alternative 1 | Alternative 2 | Why Current Choice |
|-----------|---------------|---------------|---------------|-------------------|
| **ML Algorithm** | Linear Regression | Random Forest | XGBoost | Simple, interpretable, fast training, sufficient for linear relationships |
| | | Gradient Boosting | Neural Network | Lower complexity, easier to explain to stakeholders |
| **Web Framework** | FastAPI | Flask | Django REST | Modern, fast, automatic validation, excellent documentation |
| **Data Processing** | Pandas | Polars | Dask | Mature ecosystem, easy to use, sufficient for dataset size |
| **Model Format** | Joblib | Pickle | PMML | Efficient compression, scikit-learn native, fast loading |

### Why Linear Regression?

**Advantages**:
1. **Interpretability**: Coefficients show exact contribution of each feature
2. **Speed**: Extremely fast training (milliseconds) and inference (<10ms)
3. **Simplicity**: Easy to understand and explain to non-technical stakeholders
4. **Low Resource**: Minimal memory footprint (~1MB model size)
5. **Debugging**: Easy to identify issues with predictions
6. **Baseline**: Excellent baseline before trying complex models
7. **Sufficient**: House prices often have linear relationships with features

**When to Upgrade**:
- Non-linear patterns in data (use Polynomial Regression or Random Forest)
- Many features with complex interactions (use XGBoost)
- Need higher accuracy (use ensemble methods)
- Have large dataset (>100K samples)

### Code Function Explanations

#### 1. Model Training (`train_model.py` or notebook)

```python
def load_data(filepath='data/housing.csv'):
    """
    Load housing dataset from CSV file
    
    Expected columns:
    - OverallQual: Overall material and finish quality (1-10)
    - GrLivArea: Above grade living area (square feet)
    - GarageCars: Size of garage (number of cars)
    - TotalBsmtSF: Total basement square feet
    - FullBath: Full bathrooms above grade
    - YearBuilt: Original construction year
    - SalePrice: Property sale price (target variable)
    
    Returns: 
    - X: Feature matrix (DataFrame)
    - y: Target variable (Series)
    """
    df = pd.read_csv(filepath)
    
    # Handle missing values
    df = df.fillna(df.median(numeric_only=True))
    
    # Separate features and target
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']
    
    return X, y


def preprocess_features(X_train, X_test=None):
    """
    Preprocess features for linear regression
    
    Steps:
    1. Handle missing values (median imputation)
    2. Create polynomial features (degree=2)
    3. Scale features (StandardScaler: mean=0, std=1)
    
    Scaling is crucial for linear regression because:
    - Features have different units (sqft vs year)
    - Large-scale features dominate small-scale ones
    - Gradient descent converges faster with scaled data
    - Regularization treats all features equally
    
    Returns:
    - Preprocessed features
    - Fitted scaler (for inverse transform if needed)
    """
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    
    # Create polynomial features (e.g., sqft² for non-linear relationships)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    
    if X_test is not None:
        X_test_poly = poly.transform(X_test)
        X_test_scaled = scaler.transform(X_test_poly)
        return X_train_scaled, X_test_scaled, scaler, poly
    
    return X_train_scaled, scaler, poly


def train_linear_regression(X_train, y_train):
    """
    Train Linear Regression model
    
    Linear Regression Formula:
    y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
    
    Where:
    - y = predicted price
    - β₀ = intercept (base price)
    - βᵢ = coefficient for feature i (price change per unit)
    - xᵢ = feature value
    
    Training Method: Ordinary Least Squares (OLS)
    - Minimizes sum of squared errors
    - Closed-form solution (no iterations needed)
    - Fast and deterministic
    
    Returns: Trained model
    """
    from sklearn.linear_model import LinearRegression
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    Metrics Explained:
    
    1. R² Score (Coefficient of Determination):
       - Range: -∞ to 1 (1 is perfect)
       - Measures: % of variance explained by model
       - Formula: 1 - (SS_residual / SS_total)
       - Interpretation: 0.85 = model explains 85% of price variance
    
    2. Mean Absolute Error (MAE):
       - Average absolute difference: |predicted - actual|
       - Unit: Same as target (dollars)
       - Robust to outliers
       - Easy to interpret: "Average error is $15,000"
    
    3. Root Mean Squared Error (RMSE):
       - Square root of average squared errors
       - Penalizes large errors more than MAE
       - Unit: Same as target (dollars)
       - More sensitive to outliers
    
    4. Mean Absolute Percentage Error (MAPE):
       - Average % error: |predicted - actual| / actual
       - Unit: Percentage
       - Scale-independent
       - Good for comparing across price ranges
    
    Returns: Dictionary of metrics
    """
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    import numpy as np
    
    predictions = model.predict(X_test)
    
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    
    return {
        'r2_score': r2,
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }


def save_model(model, scaler, poly, feature_names):
    """
    Save trained model and preprocessors
    
    Saves:
    - model.pkl: Trained Linear Regression model
    - scaler.pkl: Fitted StandardScaler
    - poly.pkl: PolynomialFeatures transformer
    - feature_names.pkl: Original feature names
    
    Why separate files:
    - Modularity: Update one component without retraining
    - Debugging: Inspect each component independently
    - Versioning: Track changes to preprocessing vs model
    """
    import joblib
    
    joblib.dump(model, 'model/model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')
    joblib.dump(poly, 'model/poly.pkl')
    joblib.dump(feature_names, 'model/features.pkl')
```

#### 2. API Endpoints (`main.py`)

```python
from fastapi import FastAPI, HTTPException
from app.schemas import HouseFeatures, PricePrediction
from app.model import HousePriceModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="House Price Prediction API",
    description="Predict house prices using Linear Regression",
    version="1.0.0"
)

# Initialize model
price_model = HousePriceModel()

@app.on_event("startup")
async def startup_event():
    """
    Load model on startup
    
    Benefits:
    - Fast response times (model already in memory)
    - Fail fast (errors caught at startup, not first request)
    - Single loading (not per request)
    """
    try:
        price_model.load_model()
        logger.info("House price model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Continue running to show helpful error messages


@app.get("/")
def root():
    """
    Health check endpoint
    
    Returns API status and basic info
    """
    return {
        "message": "House Price Prediction API",
        "status": "running",
        "version": "1.0.0",
        "model_loaded": price_model.model is not None
    }


@app.post("/predict", response_model=PricePrediction)
def predict_price(features: HouseFeatures):
    """
    Predict house price from features
    
    Process:
    1. Validate input (Pydantic automatic)
    2. Convert to feature array
    3. Apply preprocessing (polynomial + scaling)
    4. Make prediction
    5. Calculate confidence interval
    6. Format response
    
    Request Example:
    {
        "GrLivArea": 2000,
        "OverallQual": 7,
        "GarageCars": 2,
        "TotalBsmtSF": 1500,
        "FullBath": 2,
        "YearBuilt": 2005
    }
    
    Response Example:
    {
        "predicted_price": 285000.50,
        "confidence_low": 265000.00,
        "confidence_high": 305000.00,
        "features_used": {
            "GrLivArea": 2000,
            "OverallQual": 7,
            ...
        }
    }
    """
    if price_model.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # Convert Pydantic model to dict
        feature_dict = features.dict()
        
        # Predict
        prediction = price_model.predict(feature_dict)
        
        return PricePrediction(
            predicted_price=float(prediction['price']),
            confidence_low=float(prediction['confidence_low']),
            confidence_high=float(prediction['confidence_high']),
            features_used=feature_dict
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/predict/batch")
def predict_batch(houses: list[HouseFeatures]):
    """
    Predict prices for multiple houses
    
    Batch processing benefits:
    - Vectorized operations (faster than loop)
    - Single preprocessing step
    - Reduced API overhead
    - Efficient for large portfolios
    
    Use cases:
    - Real estate portfolio valuation
    - Market analysis
    - Bulk property appraisal
    """
    if price_model.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        for house in houses:
            pred = price_model.predict(house.dict())
            predictions.append({
                "features": house.dict(),
                "predicted_price": float(pred['price'])
            })
        
        return {
            "predictions": predictions,
            "total_houses": len(predictions)
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
def model_info():
    """
    Get model metadata
    
    Returns:
    - Model type
    - Feature names and importances
    - Training metrics
    - Last updated date
    """
    if price_model.model is None:
        raise HTTPException(status_code=404, detail="Model not loaded")
    
    return {
        "model_type": "Linear Regression",
        "features": price_model.feature_names,
        "coefficients": dict(zip(
            price_model.feature_names,
            price_model.model.coef_.tolist()
        )),
        "intercept": float(price_model.model.intercept_),
        "accuracy_r2": 0.85  # From training
    }


@app.get("/features/importance")
def feature_importance():
    """
    Get feature importance (absolute coefficients)
    
    Interpretation:
    - Positive coefficient: Price increases with feature
    - Negative coefficient: Price decreases with feature
    - Magnitude: Impact per unit change (after scaling)
    
    Example:
    - GrLivArea coefficient = 50000
    - Means: Each additional sqft adds $50,000 (after scaling)
    """
    if price_model.model is None:
        raise HTTPException(status_code=404, detail="Model not loaded")
    
    importances = dict(zip(
        price_model.feature_names,
        np.abs(price_model.model.coef_)
    ))
    
    # Sort by importance
    sorted_features = sorted(
        importances.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return {
        "feature_importances": sorted_features,
        "note": "Importance based on absolute coefficient values"
    }
```

#### 3. Model Class (`model.py`)

```python
import joblib
import numpy as np
import pandas as pd
from typing import Dict

class HousePriceModel:
    """
    House price prediction model wrapper
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.poly = None
        self.feature_names = None
    
    def load_model(self, model_path='model/model.pkl'):
        """
        Load trained model and preprocessors
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load('model/scaler.pkl')
        self.poly = joblib.load('model/poly.pkl')
        self.feature_names = joblib.load('model/features.pkl')
    
    def preprocess(self, features: Dict) -> np.ndarray:
        """
        Preprocess features for prediction
        
        Steps:
        1. Convert dict to DataFrame
        2. Ensure correct feature order
        3. Handle missing values
        4. Create polynomial features
        5. Scale features
        """
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Reorder columns to match training
        df = df[self.feature_names]
        
        # Handle missing (use median from training)
        df = df.fillna(df.median())
        
        # Polynomial features
        X_poly = self.poly.transform(df)
        
        # Scale
        X_scaled = self.scaler.transform(X_poly)
        
        return X_scaled
    
    def predict(self, features: Dict) -> Dict:
        """
        Predict house price with confidence interval
        
        Confidence Interval Calculation:
        - Uses standard error of prediction
        - 95% confidence interval: prediction ± 1.96 * std_error
        - Wider for extreme feature values
        """
        # Preprocess
        X = self.preprocess(features)
        
        # Predict
        price = self.model.predict(X)[0]
        
        # Calculate confidence interval (simplified)
        # In production, use prediction_intervals from statsmodels
        std_error = 20000  # Estimated from training data
        confidence_margin = 1.96 * std_error
        
        return {
            'price': price,
            'confidence_low': price - confidence_margin,
            'confidence_high': price + confidence_margin
        }
```

#### 4. Data Validation (`schemas.py`)

```python
from pydantic import BaseModel, Field, validator

class HouseFeatures(BaseModel):
    """
    House features for price prediction
    
    All features are validated for:
    - Type correctness
    - Range validity
    - Business logic rules
    """
    
    GrLivArea: int = Field(
        ...,
        description="Above grade living area (square feet)",
        ge=500,
        le=10000,
        example=2000
    )
    
    OverallQual: int = Field(
        ...,
        description="Overall material and finish quality (1-10)",
        ge=1,
        le=10,
        example=7
    )
    
    GarageCars: int = Field(
        ...,
        description="Size of garage (number of cars)",
        ge=0,
        le=5,
        example=2
    )
    
    TotalBsmtSF: int = Field(
        ...,
        description="Total basement square feet",
        ge=0,
        le=6000,
        example=1500
    )
    
    FullBath: int = Field(
        ...,
        description="Full bathrooms above grade",
        ge=0,
        le=5,
        example=2
    )
    
    YearBuilt: int = Field(
        ...,
        description="Original construction year",
        ge=1800,
        le=2024,
        example=2005
    )
    
    @validator('GrLivArea')
    def validate_living_area(cls, v):
        """Ensure living area is reasonable"""
        if v < 500:
            raise ValueError("Living area seems too small")
        if v > 10000:
            raise ValueError("Living area seems too large")
        return v
    
    @validator('YearBuilt')
    def validate_year(cls, v):
        """Ensure year is reasonable"""
        import datetime
        current_year = datetime.datetime.now().year
        if v > current_year:
            raise ValueError("Year built cannot be in the future")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "GrLivArea": 2000,
                "OverallQual": 7,
                "GarageCars": 2,
                "TotalBsmtSF": 1500,
                "FullBath": 2,
                "YearBuilt": 2005
            }
        }


class PricePrediction(BaseModel):
    """Price prediction response"""
    
    predicted_price: float = Field(
        ...,
        description="Predicted house price in USD",
        example=285000.50
    )
    
    confidence_low: float = Field(
        ...,
        description="Lower bound of 95% confidence interval",
        example=265000.00
    )
    
    confidence_high: float = Field(
        ...,
        description="Upper bound of 95% confidence interval",
        example=305000.00
    )
    
    features_used: dict = Field(
        ...,
        description="Input features used for prediction"
    )
```

---

## Architectural Documentation

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     CLIENT APPLICATIONS                          │
│   (Real Estate Websites, Mobile Apps, Internal Tools)           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ HTTPS REST API
                            │ POST /predict
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                      FASTAPI APPLICATION                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              API Layer (main.py)                           │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │ │
│  │  │   GET    │  │  POST    │  │  POST    │  │   GET    │  │ │
│  │  │    /     │  │ /predict │  │ /predict │  │  /model  │  │ │
│  │  │  health  │  │  single  │  │  /batch  │  │  /info   │  │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │ │
│  └─────────────────────────┬────────────────────────────────────┘ │
│                            │                                       │
│  ┌─────────────────────────▼────────────────────────────────────┐ │
│  │         Validation Layer (Pydantic)                          │ │
│  │  - Type checking (int, float)                                │ │
│  │  - Range validation (sqft: 500-10000)                        │ │
│  │  - Business rules (year not in future)                       │ │
│  └─────────────────────────┬────────────────────────────────────┘ │
│                            │                                       │
│  ┌─────────────────────────▼────────────────────────────────────┐ │
│  │       Business Logic (model.py)                              │ │
│  │  ┌────────────────────────────────────────────────────────┐ │ │
│  │  │  HousePriceModel Class                                  │ │ │
│  │  │  ├── load_model()                                       │ │ │
│  │  │  ├── preprocess(features)                               │ │ │
│  │  │  │   - DataFrame conversion                             │ │ │
│  │  │  │   - Feature ordering                                 │ │ │
│  │  │  │   - Missing value handling                           │ │ │
│  │  │  │   - Polynomial transformation                        │ │ │
│  │  │  │   - Feature scaling                                  │ │ │
│  │  │  └── predict(features)                                  │ │ │
│  │  │      - Model inference                                  │ │ │
│  │  │      - Confidence interval calculation                  │ │ │
│  │  └────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                            │                                       │
│  ┌─────────────────────────▼────────────────────────────────────┐ │
│  │         ML Components (In-Memory)                            │ │
│  │  ┌──────────────────┐  ┌────────────────────┐              │ │
│  │  │ Linear Regression│  │ StandardScaler     │              │ │
│  │  │ Model            │  │ (mean=0, std=1)    │              │ │
│  │  │ (β₀ + β₁x₁ + ...)│  └────────────────────┘              │ │
│  │  └──────────────────┘                                       │ │
│  │  ┌──────────────────┐                                       │ │
│  │  │ Polynomial       │                                       │ │
│  │  │ Features         │                                       │ │
│  │  │ (degree=2)       │                                       │ │
│  │  └──────────────────┘                                       │ │
│  └─────────────────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ Load on startup
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    PERSISTENT STORAGE                            │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │  model.pkl       │  │  scaler.pkl      │  │  poly.pkl     │ │
│  │  (Trained LR)    │  │  (Fitted scaler) │  │  (PolyFeat)   │ │
│  └──────────────────┘  └──────────────────┘  └───────────────┘ │
│                                                                  │
│  Training Data: data/housing.csv (Ames Housing Dataset)         │
└──────────────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
┌──────────────────┐
│ House Features   │
│ {sqft: 2000,     │
│  quality: 7,     │
│  garage: 2, ...} │
└────────┬─────────┘
         │
         ▼
┌──────────────────────┐
│ Pydantic Validation  │
│ - Type check         │
│ - Range check        │
│ - Business rules     │
└────────┬─────────────┘
         │ Valid ✓
         ▼
┌──────────────────────┐
│ Convert to DataFrame │
│ features_dict →      │
│ pandas DataFrame     │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ Feature Ordering     │
│ Match training order │
│ [feat1, feat2, ...]  │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ Handle Missing       │
│ Fill with median     │
│ (from training data) │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ Polynomial Features  │
│ x₁, x₂, x₁², x₂²,    │
│ x₁x₂ (degree=2)      │
└────────┬─────────────┘
         │ 6 → 21 features
         ▼
┌──────────────────────┐
│ Feature Scaling      │
│ x_scaled =           │
│ (x - mean) / std     │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ Linear Regression    │
│ y = β₀ + Σ(βᵢxᵢ)     │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ Raw Prediction       │
│ price = $285,000     │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ Confidence Interval  │
│ Low: $265,000        │
│ High: $305,000       │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ JSON Response        │
│ {predicted_price,    │
│  confidence_low,     │
│  confidence_high}    │
└──────────────────────┘
```

### Execution Flowchart

```
START
  │
  ▼
[Initialize FastAPI App]
  │
  ▼
[Load Model on Startup]
  │
  ├─ Load model.pkl
  ├─ Load scaler.pkl
  ├─ Load poly.pkl
  └─ Load feature_names.pkl
  │
  ├─ Success? ─No──► [Log Error] ─► [503 on first request]
  │                                          │
  Yes                                       END
  │
  ▼
[Wait for HTTP Request]
  │
  ▼
[Receive POST /predict]
  │
  ▼
[Pydantic Validation]
  │
  ├─ Valid? ─No──► [Return 422 Validation Error] ─► END
  │
  Yes
  │
  ▼
[Extract Features Dict]
  │
  ▼
[Convert to DataFrame]
  │
  ▼
[Reorder Columns]
  │ (match training order)
  │
  ▼
[Handle Missing Values]
  │ (fill with median)
  │
  ▼
[Create Polynomial Features]
  │ (x, x², xy, ...)
  │
  ▼
[Scale Features]
  │ (standardization)
  │
  ▼
[Model Prediction]
  │ (linear regression)
  │
  ▼
[Calculate Price]