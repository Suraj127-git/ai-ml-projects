# 📖 AI/ML Projects Technical Guide

A comprehensive technical guide for software engineers working with the AI/ML Projects collection. This document covers system architecture, implementation details, and advanced usage patterns.

## 🎯 Table of Contents

- [System Architecture](#-system-architecture)
- [Data Flow & Processing Pipeline](#-data-flow--processing-pipeline)
- [Service Architecture Patterns](#-service-architecture-patterns)
- [Model Management](#-model-management)
- [API Design Standards](#-api-design-standards)
- [Training & Inference Procedures](#-training--inference-procedures)
- [Performance Optimization](#-performance-optimization)
- [Monitoring & Observability](#-monitoring--observability)
- [Security Considerations](#-security-considerations)
- [Deployment Strategies](#-deployment-strategies)
- [Troubleshooting Guide](#-troubleshooting-guide)
- [Model Versioning](#-model-versioning)
- [References & Resources](#-references--resources)

## 🏗️ System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Gateway                              │
├─────────────────────────────────────────────────────────────────┤
│                    Load Balancer / Proxy                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │Inventory    │ │Sentiment    │ │Image        │ │Recommendation│ │
│  │Optimization │ │Analysis     │ │Classification│ │System       │ │
│  │Service      │ │Service      │ │Service      │ │Service      │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
│                                                                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │Churn        │ │Fraud        │ │Demand       │ │Chatbot      │ │
│  │Prediction   │ │Detection    │ │Forecasting  │ │API          │ │
│  │Service      │ │Service      │ │Service      │ │Service      │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                    Shared Infrastructure                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │Model Registry │ │Feature Store │ │Monitoring    │ │Logging      │ │
│  │(MLflow)     │ │(Redis)       │ │(Prometheus)  │ │(ELK Stack)  │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Microservice Architecture Pattern

Each service follows a consistent microservice pattern:

```
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Application                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│  │   API Routes    │ │  Pydantic       │ │   Middleware    │ │
│  │   & Endpoints   │ │  Validation     │ │   & CORS        │ │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘ │
│                                                                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│  │   ML Model      │ │  Data           │ │   Business      │ │
│  │   Logic         │ │  Processing     │ │   Logic         │ │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘ │
│                                                                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│  │   Model         │ │  Feature        │ │   Preprocessing │ │
│  │   Loading       │ │  Engineering    │ │   Pipeline      │ │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘ │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                    Model Storage                                │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│  │   Trained       │ │  Preprocessor   │ │   Configuration │ │
│  │   Models        │ │  Objects        │ │   Files         │ │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 Data Flow & Processing Pipeline

### Request Processing Flow

```
Client Request → API Gateway → Load Balancer → Service Instance
     ↓
FastAPI Application
     ↓
Input Validation (Pydantic)
     ↓
Preprocessing Pipeline
     ↓
Feature Engineering
     ↓
Model Inference
     ↓
Post-processing
     ↓
Response Formatting
     ↓
Client Response
```

### Batch Processing Flow

```
Batch Request → Request Splitter → Parallel Processing → Result Aggregator
     ↓              ↓                    ↓                    ↓
Validation  →  Worker Pool  →  Model Inference  →  Response Builder
     ↓              ↓                    ↓                    ↓
Preprocessing → Async Tasks  →  Result Queue   →  Batch Response
```

### Data Pipeline Components

#### 1. Input Validation Layer
```python
from pydantic import BaseModel, validator
from typing import List, Optional

class PredictionRequest(BaseModel):
    features: List[float]
    model_version: Optional[str] = "latest"
    
    @validator('features')
    def validate_features(cls, v):
        if len(v) != EXPECTED_FEATURE_COUNT:
            raise ValueError(f"Expected {EXPECTED_FEATURE_COUNT} features")
        return v
```

#### 2. Preprocessing Pipeline
```python
class PreprocessingPipeline:
    def __init__(self, scaler_path: str, encoder_path: str):
        self.scaler = joblib.load(scaler_path)
        self.encoder = joblib.load(encoder_path)
    
    def transform(self, raw_data: pd.DataFrame) -> np.ndarray:
        # Handle missing values
        data = self.handle_missing_values(raw_data)
        
        # Scale numerical features
        numerical_features = self.scaler.transform(data[numerical_cols])
        
        # Encode categorical features
        categorical_features = self.encoder.transform(data[categorical_cols])
        
        # Combine features
        return np.hstack([numerical_features, categorical_features])
```

#### 3. Feature Engineering
```python
class FeatureEngineer:
    def __init__(self):
        self.feature_definitions = {
            'ratios': self.calculate_ratios,
            'interactions': self.create_interactions,
            'polynomials': self.create_polynomial_features
        }
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        for feature_type, feature_func in self.feature_definitions.items():
            data = feature_func(data)
        return data
```

## 🏛️ Service Architecture Patterns

### Standard Service Structure

```python
# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import uvicorn

from .model import MLModel
from .schemas import PredictionRequest, PredictionResponse
from .utils import setup_logging, validate_input

app = FastAPI(
    title="ML Service API",
    description="Production ML microservice",
    version="1.0.0"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
model = MLModel()

@app.on_event("startup")
async def startup_event():
    """Initialize model and resources"""
    await model.load()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources"""
    await model.cleanup()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model.is_loaded()}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Main prediction endpoint"""
    try:
        # Validate input
        validated_data = validate_input(request)
        
        # Make prediction
        prediction = await model.predict(validated_data)
        
        return PredictionResponse(
            prediction=prediction,
            confidence=model.get_confidence(),
            model_version=model.version
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(requests: List[PredictionRequest]):
    """Batch prediction endpoint"""
    try:
        predictions = await model.predict_batch([req.features for req in requests])
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Model Class Pattern

```python
# app/model.py
import joblib
import numpy as np
from typing import List, Optional
import asyncio
from pathlib import Path

class MLModel:
    def __init__(self, model_path: str = "models/model.pkl"):
        self.model = None
        self.preprocessor = None
        self.version = "1.0.0"
        self.model_path = Path(model_path)
        self._loaded = False
    
    async def load(self):
        """Load model and preprocessor"""
        try:
            self.model = joblib.load(self.model_path)
            self.preprocessor = joblib.load(self.model_path.parent / "preprocessor.pkl")
            self._loaded = True
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    async def predict(self, features: np.ndarray) -> float:
        """Single prediction"""
        if not self._loaded:
            raise RuntimeError("Model not loaded")
        
        # Preprocess
        processed_features = self.preprocessor.transform(features.reshape(1, -1))
        
        # Predict
        prediction = self.model.predict(processed_features)[0]
        
        return float(prediction)
    
    async def predict_batch(self, features_list: List[np.ndarray]) -> List[float]:
        """Batch prediction with parallel processing"""
        if not self._loaded:
            raise RuntimeError("Model not loaded")
        
        # Convert to numpy array
        features_array = np.array(features_list)
        
        # Preprocess
        processed_features = self.preprocessor.transform(features_array)
        
        # Predict
        predictions = self.model.predict(processed_features)
        
        return predictions.tolist()
    
    def is_loaded(self) -> bool:
        return self._loaded
    
    async def cleanup(self):
        """Cleanup resources"""
        self.model = None
        self.preprocessor = None
        self._loaded = False
```

## 🧠 Model Management

### Model Registry Integration

```python
# utils/model_registry.py
import mlflow
import mlflow.sklearn
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.tracking.MlflowClient()
    
    def register_model(self, model_name: str, model_uri: str, metrics: Dict[str, float]):
        """Register model with MLflow"""
        try:
            # Register model
            model_version = mlflow.register_model(model_uri, model_name)
            
            # Add tags
            self.client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key="metrics",
                value=str(metrics)
            )
            
            logger.info(f"Model {model_name} v{model_version.version} registered")
            return model_version
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    def get_latest_model(self, model_name: str, stage: str = "Production"):
        """Get latest model from registry"""
        try:
            latest_version = self.client.get_latest_versions(model_name, stages=[stage])
            if latest_version:
                return latest_version[0]
            return None
        except Exception as e:
            logger.error(f"Failed to get latest model: {e}")
            return None
    
    def transition_model(self, model_name: str, version: int, stage: str):
        """Transition model to new stage"""
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            logger.info(f"Model {model_name} v{version} transitioned to {stage}")
        except Exception as e:
            logger.error(f"Failed to transition model: {e}")
            raise
```

### Model Versioning Strategy

```python
# utils/model_versioning.py
from typing import Dict, Any
import semver
import json
from pathlib import Path

class ModelVersioning:
    def __init__(self, version_file: str = "model_versions.json"):
        self.version_file = Path(version_file)
        self.versions = self._load_versions()
    
    def _load_versions(self) -> Dict[str, Any]:
        """Load version history"""
        if self.version_file.exists():
            with open(self.version_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_versions(self):
        """Save version history"""
        with open(self.version_file, 'w') as f:
            json.dump(self.versions, f, indent=2)
    
    def bump_version(self, model_name: str, change_type: str = "patch") -> str:
        """Bump model version"""
        current_version = self.versions.get(model_name, "1.0.0")
        
        if change_type == "major":
            new_version = semver.bump_major(current_version)
        elif change_type == "minor":
            new_version = semver.bump_minor(current_version)
        else:  # patch
            new_version = semver.bump_patch(current_version)
        
        self.versions[model_name] = new_version
        self._save_versions()
        
        return new_version
    
    def get_version_info(self, model_name: str) -> Dict[str, Any]:
        """Get version information"""
        version = self.versions.get(model_name)
        if version:
            return {
                "version": version,
                "major": semver.parse(version)["major"],
                "minor": semver.parse(version)["minor"],
                "patch": semver.parse(version)["patch"]
            }
        return None
```

## 📐 API Design Standards

### RESTful API Guidelines

#### 1. Resource Naming
```
✅ Good:
GET /predictions
POST /predictions
GET /predictions/{id}
PUT /predictions/{id}
DELETE /predictions/{id}

❌ Bad:
GET /getPredictions
POST /createPrediction
GET /predictionData/{id}
```

#### 2. HTTP Status Codes
```python
# Success responses
200 OK - Successful GET, PUT
201 Created - Successful POST
202 Accepted - Request accepted for processing
204 No Content - Successful DELETE

# Client error responses
400 Bad Request - Invalid request data
401 Unauthorized - Missing authentication
403 Forbidden - Insufficient permissions
404 Not Found - Resource not found
409 Conflict - Resource conflict
422 Unprocessable Entity - Validation error

# Server error responses
500 Internal Server Error - Server error
502 Bad Gateway - Invalid upstream response
503 Service Unavailable - Service unavailable
504 Gateway Timeout - Upstream timeout
```

#### 3. Response Format Standards
```python
# Success response
{
    "status": "success",
    "data": {
        "prediction": 0.85,
        "confidence": 0.92,
        "model_version": "1.0.0"
    },
    "timestamp": "2024-01-15T10:30:00Z"
}

# Error response
{
    "status": "error",
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "Invalid input format",
        "details": {
            "field": "features",
            "issue": "Expected 10 features, got 8"
        }
    },
    "timestamp": "2024-01-15T10:30:00Z"
}
```

### API Versioning Strategy

```python
# Header-based versioning
@app.get("/predict")
async def predict(
    request: PredictionRequest,
    api_version: str = Header("1.0", alias="API-Version")
):
    if api_version == "1.0":
        return await predict_v1(request)
    elif api_version == "2.0":
        return await predict_v2(request)
    else:
        raise HTTPException(status_code=400, detail="Unsupported API version")

# URL-based versioning
@app.get("/v1/predict")
async def predict_v1(request: PredictionRequest):
    # V1 implementation
    pass

@app.get("/v2/predict")
async def predict_v2(request: PredictionRequestV2):
    # V2 implementation
    pass

### Advanced API Usage Examples

#### 1. Multi-Service Orchestration
```python
# services/orchestrator.py
import asyncio
import aiohttp
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class MLServiceOrchestrator:
    def __init__(self, service_urls: Dict[str, str]):
        self.service_urls = service_urls
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def analyze_customer_insights(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate multiple ML services for comprehensive customer analysis."""
        
        # Prepare concurrent requests
        tasks = []
        
        # Churn prediction
        churn_task = self._call_service(
            "churn_prediction",
            "/predict",
            customer_data
        )
        tasks.append(("churn_risk", churn_task))
        
        # Customer lifetime value
        clv_task = self._call_service(
            "clv_predictor",
            "/predict",
            customer_data
        )
        tasks.append(("lifetime_value", clv_task))
        
        # Sentiment analysis (if customer has reviews)
        if "recent_reviews" in customer_data:
            sentiment_task = self._call_service(
                "sentiment_analysis",
                "/analyze",
                {"text": " ".join(customer_data["recent_reviews"])}
            )
            tasks.append(("sentiment", sentiment_task))
        
        # Execute all requests concurrently
        results = {}
        for key, task in tasks:
            try:
                results[key] = await task
            except Exception as e:
                logger.error(f"Service call failed for {key}: {e}")
                results[key] = None
        
        # Combine results
        return self._combine_insights(results)
    
    async def _call_service(self, service_name: str, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific ML service."""
        url = f"{self.service_urls[service_name]}{endpoint}"
        
        async with self.session.post(url, json=data) as response:
            response.raise_for_status()
            return await response.json()
    
    def _combine_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine insights from multiple services."""
        insights = {
            "customer_risk_profile": {},
            "recommended_actions": [],
            "confidence_score": 0.0
        }
        
        # Churn risk analysis
        if results.get("churn_risk"):
            churn_data = results["churn_risk"]
            insights["customer_risk_profile"]["churn_probability"] = churn_data.get("churn_probability", 0)
            insights["customer_risk_profile"]["risk_level"] = churn_data.get("risk_level", "unknown")
            
            if churn_data.get("churn_probability", 0) > 0.7:
                insights["recommended_actions"].append("Implement retention strategy")
        
        # Customer lifetime value
        if results.get("lifetime_value"):
            clv_data = results["lifetime_value"]
            insights["customer_risk_profile"]["predicted_clv"] = clv_data.get("predicted_clv", 0)
            insights["customer_risk_profile"]["clv_tier"] = clv_data.get("clv_tier", "unknown")
        
        # Sentiment analysis
        if results.get("sentiment"):
            sentiment_data = results["sentiment"]
            insights["customer_risk_profile"]["sentiment_score"] = sentiment_data.get("confidence", 0)
            insights["customer_risk_profile"]["overall_sentiment"] = sentiment_data.get("sentiment", "neutral")
        
        # Calculate overall confidence
        confidence_scores = []
        for key in ["churn_risk", "lifetime_value", "sentiment"]:
            if results.get(key) and results[key].get("confidence"):
                confidence_scores.append(results[key]["confidence"])
        
        insights["confidence_score"] = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        return insights

# Usage example
async def main():
    service_urls = {
        "churn_prediction": "http://localhost:8007",
        "clv_predictor": "http://localhost:8008",
        "sentiment_analysis": "http://localhost:8004"
    }
    
    customer_data = {
        "customer_id": "CUST001",
        "tenure_months": 24,
        "monthly_charges": 65.50,
        "total_charges": 1572.00,
        "contract_type": "month-to-month",
        "service_issues": 2,
        "recent_reviews": [
            "Great service, highly recommend!",
            "Customer support was very helpful"
        ]
    }
    
    async with MLServiceOrchestrator(service_urls) as orchestrator:
        insights = await orchestrator.analyze_customer_insights(customer_data)
        print(f"Customer Insights: {insights}")

if __name__ == "__main__":
    asyncio.run(main())
```

#### 2. Error Handling and Circuit Breaker Pattern
```python
# utils/circuit_breaker.py
import time
import threading
from typing import Callable, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open" # Testing if service recovered

class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self._failure_count = 0
        self._last_failure_time = None
        self._state = CircuitState.CLOSED
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker: attempting reset")
                else:
                    raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return (
            self._last_failure_time and
            time.time() - self._last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            self._failure_count = 0
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                logger.info("Circuit breaker: CLOSED")
    
    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.error(f"Circuit breaker: OPEN (failures: {self._failure_count})")
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit breaker state."""
        return self._state

# Usage with ML services
import requests
from typing import Dict, Any

def call_ml_service(url: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Call ML service with circuit breaker protection."""
    response = requests.post(url, json=data, timeout=30)
    response.raise_for_status()
    return response.json()

# Create circuit breaker for each service
service_breakers = {
    "inventory_optimization": CircuitBreaker(failure_threshold=3, recovery_timeout=30),
    "sentiment_analysis": CircuitBreaker(failure_threshold=5, recovery_timeout=60),
    "fraud_detection": CircuitBreaker(failure_threshold=2, recovery_timeout=15)
}

def protected_service_call(service_name: str, url: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Make protected service call with circuit breaker."""
    breaker = service_breakers.get(service_name)
    if not breaker:
        return call_ml_service(url, data)
    
    try:
        return breaker.call(call_ml_service, url, data)
    except Exception as e:
        logger.error(f"Service call failed for {service_name}: {e}")
        return None

# Example usage
result = protected_service_call(
    "inventory_optimization",
    "http://localhost:8003/optimize",
    {"product_id": "PROD001", "unit_cost": 25.0, "demand_rate": 500}
)
```

#### 3. Rate Limiting and Throttling
```python
# utils/rate_limiter.py
import time
import threading
from typing import Dict, Callable
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, rate: int = 10, burst: int = 20):
        self.rate = rate  # requests per second
        self.burst = burst
        self.tokens = burst
        self.last_update = time.time()
        self.lock = threading.Lock()
    
    def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens, return True if successful."""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            
            # Add tokens based on elapsed time
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def wait_time(self, tokens: int = 1) -> float:
        """Calculate wait time for tokens."""
        with self.lock:
            if self.acquire(tokens):
                return 0.0
            
            return (tokens - self.tokens) / self.rate

class SlidingWindowRateLimiter:
    """Sliding window rate limiter for per-client limiting."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(deque)
        self.lock = threading.Lock()
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client."""
        with self.lock:
            now = time.time()
            minute_ago = now - 60
            
            # Remove old requests
            while self.requests[client_id] and self.requests[client_id][0] <= minute_ago:
                self.requests[client_id].popleft()
            
            # Check if under limit
            if len(self.requests[client_id]) < self.requests_per_minute:
                self.requests[client_id].append(now)
                return True
            
            return False
    
    def get_wait_time(self, client_id: str) -> float:
        """Get wait time until next request is allowed."""
        with self.lock:
            if self.is_allowed(client_id):
                return 0.0
            
            if not self.requests[client_id]:
                return 0.0
            
            # Wait until oldest request is outside window
            oldest_request = self.requests[client_id][0]
            return max(0, 60 - (time.time() - oldest_request))

# Usage with FastAPI
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# Global rate limiter
rate_limiter = SlidingWindowRateLimiter(requests_per_minute=60)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware."""
    client_id = request.client.host
    
    if not rate_limiter.is_allowed(client_id):
        wait_time = rate_limiter.get_wait_time(client_id)
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "wait_time_seconds": wait_time,
                "retry_after": wait_time
            },
            headers={"Retry-After": str(int(wait_time))}
        )
    
    response = await call_next(request)
    return response

# Service-specific rate limiting
def rate_limited_service_call(
    service_name: str,
    url: str,
    data: Dict[str, Any],
    max_calls_per_second: int = 5
) -> Dict[str, Any]:
    """Make rate-limited service call."""
    limiter = RateLimiter(rate=max_calls_per_second, burst=max_calls_per_second * 2)
    
    # Wait for token
    while not limiter.acquire():
        wait_time = limiter.wait_time()
        logger.info(f"Rate limited, waiting {wait_time:.2f}s")
        time.sleep(wait_time)
    
    # Make actual call
    response = requests.post(url, json=data, timeout=30)
    response.raise_for_status()
    return response.json()
```

## 🚀 Training & Inference Procedures

### Model Training Pipeline

```python
# notebooks/train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path

class ModelTrainingPipeline:
    def __init__(self, data_path: str, model_output_path: str):
        self.data_path = Path(data_path)
        self.model_output_path = Path(model_output_path)
        self.model = None
        self.preprocessor = None
        
    def load_data(self) -> pd.DataFrame:
        """Load and validate training data"""
        data = pd.read_csv(self.data_path)
        
        # Basic validation
        if data.empty:
            raise ValueError("Training data is empty")
        
        if 'target' not in data.columns:
            raise ValueError("Target column not found")
        
        return data
    
    def preprocess_data(self, data: pd.DataFrame) -> tuple:
        """Preprocess data for training"""
        # Separate features and target
        X = data.drop('target', axis=1)
        y = data['target']
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train the model"""
        # Initialize model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Train with MLflow tracking
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 10)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            mlflow.log_metric("cv_mean", cv_scores.mean())
            mlflow.log_metric("cv_std", cv_scores.std())
            
            self.model = model
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Evaluate model performance"""
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", report["accuracy"])
        mlflow.log_metric("precision", report["weighted avg"]["precision"])
        mlflow.log_metric("recall", report["weighted avg"]["recall"])
        mlflow.log_metric("f1_score", report["weighted avg"]["f1-score"])
        
        return report, cm
    
    def save_model(self):
        """Save trained model and artifacts"""
        # Create output directory
        self.model_output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, self.model_output_path / "model.pkl")
        
        # Save to MLflow
        mlflow.sklearn.log_model(self.model, "model")
        
        print(f"Model saved to {self.model_output_path}")
    
    def run_training(self):
        """Run complete training pipeline"""
        # Load data
        data = self.load_data()
        
        # Preprocess
        X_train, X_test, y_train, y_test = self.preprocess_data(data)
        
        # Train model
        self.train_model(X_train, y_train)
        
        # Evaluate
        report, cm = self.evaluate_model(X_test, y_test)
        
        # Save model
        self.save_model()
        
        return report, cm

if __name__ == "__main__":
    pipeline = ModelTrainingPipeline(
        data_path="data/training_data.csv",
        model_output_path="models/"
    )
    
    report, cm = pipeline.run_training()
    print("Training completed successfully!")
```

### Inference Optimization

```python
# utils/inference_optimizer.py
import numpy as np
import onnxruntime as ort
import torch
import tensorflow as tf
from typing import Union, List
import time

class InferenceOptimizer:
    def __init__(self, model_path: str, optimization_level: str = "O1"):
        self.model_path = model_path
        self.optimization_level = optimization_level
        self.session = None
        self.model_type = self._detect_model_type()
        
    def _detect_model_type(self) -> str:
        """Detect model type from file extension"""
        if self.model_path.endswith('.onnx'):
            return 'onnx'
        elif self.model_path.endswith('.pt') or self.model_path.endswith('.pth'):
            return 'pytorch'
        elif self.model_path.endswith('.h5') or self.model_path.endswith('.pb'):
            return 'tensorflow'
        else:
            return 'sklearn'
    
    def optimize_model(self):
        """Optimize model for inference"""
        if self.model_type == 'onnx':
            return self._optimize_onnx()
        elif self.model_type == 'pytorch':
            return self._optimize_pytorch()
        elif self.model_type == 'tensorflow':
            return self._optimize_tensorflow()
        else:
            return self._optimize_sklearn()
    
    def _optimize_onnx(self):
        """Optimize ONNX model"""
        # Create ONNX Runtime session with optimization
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        
        # Enable graph optimization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        return self.session
    
    def _optimize_pytorch(self):
        """Optimize PyTorch model"""
        model = torch.load(self.model_path)
        model.eval()
        
        # Enable inference mode
        with torch.inference_mode():
            # JIT compilation for optimization
            if self.optimization_level == "O2":
                model = torch.jit.script(model)
            elif self.optimization_level == "O1":
                model = torch.jit.trace(model, example_inputs=torch.randn(1, 10))
        
        return model
    
    def predict_optimized(self, input_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Make optimized prediction"""
        start_time = time.time()
        
        if self.model_type == 'onnx':
            input_name = self.session.get_inputs()[0].name
            prediction = self.session.run(None, {input_name: input_data})[0]
        
        elif self.model_type == 'pytorch':
            with torch.inference_mode():
                if isinstance(input_data, np.ndarray):
                    input_data = torch.from_numpy(input_data).float()
                prediction = self.model(input_data).numpy()
        
        inference_time = time.time() - start_time
        return prediction, inference_time
```

## ⚡ Performance Optimization

### Caching Strategies

```python
# utils/caching.py
import redis
import pickle
from functools import wraps
from typing import Any, Optional
import hashlib
import json

class ModelCache:
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
        self.default_ttl = 3600  # 1 hour
    
    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generate cache key from data"""
        data_str = json.dumps(data, sort_keys=True)
        data_hash = hashlib.md5(data_str.encode()).hexdigest()
        return f"{prefix}:{data_hash}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        try:
            cached_value = self.redis_client.get(key)
            if cached_value:
                return pickle.loads(cached_value)
        except Exception as e:
            print(f"Cache get error: {e}")
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set cached value"""
        try:
            serialized_value = pickle.dumps(value)
            self.redis_client.setex(
                key, 
                ttl or self.default_ttl, 
                serialized_value
            )
        except Exception as e:
            print(f"Cache set error: {e}")
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate cache by pattern"""
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
        except Exception as e:
            print(f"Cache invalidation error: {e}")

def cached_prediction(cache: ModelCache, ttl: int = 3600):
    """Decorator for caching predictions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Generate cache key
            cache_key = cache._generate_key(
                f"prediction:{func.__name__}", 
                {"args": args, "kwargs": kwargs}
            )
            
            # Check cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(self, *args, **kwargs)
            
            # Cache result
            cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator
```

### Batch Processing Optimization

```python
# utils/batch_processor.py
import asyncio
import aiohttp
from typing import List, Callable, Any
import time
from concurrent.futures import ThreadPoolExecutor

class BatchProcessor:
    def __init__(self, max_workers: int = 4, batch_size: int = 32):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_batch(self, items: List[Any], processor_func: Callable) -> List[Any]:
        """Process items in batches"""
        results = []
        
        # Split into batches
        batches = [items[i:i + self.batch_size] 
                  for i in range(0, len(items), self.batch_size)]
        
        # Process batches in parallel
        tasks = []
        for batch in batches:
            task = asyncio.create_task(self._process_single_batch(batch, processor_func))
            tasks.append(task)
        
        # Collect results
        batch_results = await asyncio.gather(*tasks)
        for batch_result in batch_results:
            results.extend(batch_result)
        
        return results
    
    async def _process_single_batch(self, batch: List[Any], processor_func: Callable) -> List[Any]:
        """Process a single batch"""
        loop = asyncio.get_event_loop()
        
        # Run processor function in thread pool
        results = await loop.run_in_executor(
            self.executor, 
            processor_func, 
            batch
        )
        
        return results
    
    def optimize_batch_size(self, sample_items: List[Any], processor_func: Callable) -> int:
        """Find optimal batch size through benchmarking"""
        batch_sizes = [8, 16, 32, 64, 128]
        optimal_size = self.batch_size
        best_time = float('inf')
        
        for batch_size in batch_sizes:
            start_time = time.time()
            
            # Process sample with current batch size
            batches = [sample_items[i:i + batch_size] 
                      for i in range(0, len(sample_items), batch_size)]
            
            for batch in batches:
                processor_func(batch)
            
            processing_time = time.time() - start_time
            
            if processing_time < best_time:
                best_time = processing_time
                optimal_size = batch_size
        
        return optimal_size
```

## 📊 Monitoring & Observability

### Metrics Collection

```python
# utils/metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time
from functools import wraps
from typing import Callable

# Define metrics
prediction_counter = Counter(
    'ml_predictions_total', 
    'Total number of predictions',
    ['model_name', 'status']
)

prediction_latency = Histogram(
    'ml_prediction_latency_seconds',
    'Prediction latency in seconds',
    ['model_name']
)

model_accuracy = Gauge(
    'ml_model_accuracy',
    'Current model accuracy',
    ['model_name', 'version']
)

active_requests = Gauge(
    'ml_active_requests',
    'Number of active requests',
    ['model_name']
)

def track_metrics(model_name: str):
    """Decorator for tracking prediction metrics"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            active_requests.labels(model_name=model_name).inc()
            
            try:
                result = await func(*args, **kwargs)
                prediction_counter.labels(
                    model_name=model_name, 
                    status="success"
                ).inc()
                return result
            
            except Exception as e:
                prediction_counter.labels(
                    model_name=model_name, 
                    status="error"
                ).inc()
                raise e
            
            finally:
                duration = time.time() - start_time
                prediction_latency.labels(model_name=model_name).observe(duration)
                active_requests.labels(model_name=model_name).dec()
        
        return wrapper
    return decorator

# Add metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()
```

### Structured Logging

```python
# utils/logging_config.py
import logging
import json
from datetime import datetime
from typing import Dict, Any
import sys

class StructuredLogger:
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Create console handler with JSON formatter
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(self._get_json_formatter())
        self.logger.addHandler(handler)
    
    def _get_json_formatter(self):
        """JSON formatter for structured logging"""
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno
                }
                
                # Add extra fields
                if hasattr(record, 'request_id'):
                    log_entry['request_id'] = record.request_id
                if hasattr(record, 'user_id'):
                    log_entry['user_id'] = record.user_id
                if hasattr(record, 'model_name'):
                    log_entry['model_name'] = record.model_name
                if hasattr(record, 'prediction_id'):
                    log_entry['prediction_id'] = record.prediction_id
                
                return json.dumps(log_entry)
        
        return JsonFormatter()
    
    def info(self, message: str, **kwargs):
        """Info level logging"""
        for key, value in kwargs.items():
            setattr(self.logger, key, value)
        self.logger.info(message)
    
    def error(self, message: str, **kwargs):
        """Error level logging"""
        for key, value in kwargs.items():
            setattr(self.logger, key, value)
        self.logger.error(message)
    
    def debug(self, message: str, **kwargs):
        """Debug level logging"""
        for key, value in kwargs.items():
            setattr(self.logger, key, value)
        self.logger.debug(message)

# Usage example
logger = StructuredLogger("ml_service")
logger.info("Model loaded successfully", model_name="inventory_optimization", version="1.0.0")
```

### Health Checks and Readiness Probes

```python
# utils/health_checks.py
from fastapi import APIRouter, HTTPException
import psutil
import asyncio
from typing import Dict, Any
import aiohttp

router = APIRouter(prefix="/health", tags=["health"])

class HealthChecker:
    def __init__(self):
        self.checks = {
            "database": self.check_database,
            "model": self.check_model,
            "memory": self.check_memory,
            "disk": self.check_disk_space
        }
    
    async def check_database(self) -> Dict[str, Any]:
        """Check database connectivity"""
        try:
            # Simulate database check
            await asyncio.sleep(0.1)
            return {"status": "healthy", "latency_ms": 100}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def check_model(self) -> Dict[str, Any]:
        """Check model availability"""
        try:
            # Check if model files exist and are accessible
            # This would be implemented based on your model storage
            return {"status": "healthy", "model_loaded": True}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def check_memory(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            return {
                "status": "healthy" if memory.percent < 90 else "warning",
                "usage_percent": memory.percent,
                "available_gb": memory.available / (1024**3)
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def check_disk_space(self) -> Dict[str, Any]:
        """Check disk space"""
        try:
            disk = psutil.disk_usage('/')
            return {
                "status": "healthy" if disk.percent < 85 else "warning",
                "usage_percent": disk.percent,
                "free_gb": disk.free / (1024**3)
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

health_checker = HealthChecker()

@router.get("/live")
async def liveness():
    """Kubernetes liveness probe"""
    return {"status": "alive", "service": "ml-service"}

@router.get("/ready")
async def readiness():
    """Kubernetes readiness probe"""
    results = {}
    
    # Run all health checks
    for check_name, check_func in health_checker.checks.items():
        results[check_name] = await check_func()
    
    # Determine overall status
    overall_status = "healthy"
    for check_result in results.values():
        if check_result["status"] == "unhealthy":
            overall_status = "unhealthy"
            break
        elif check_result["status"] == "warning":
            overall_status = "warning"
    
    if overall_status == "unhealthy":
        raise HTTPException(status_code=503, detail={"status": "unhealthy", "checks": results})
    
    return {"status": overall_status, "checks": results}

@router.get("/startup")
async def startup():
    """Startup probe"""
    # Check if service is ready to accept requests
    try:
        # Add startup checks here
        return {"status": "ready", "service": "ml-service"}
    except Exception as e:
        raise HTTPException(status_code=503, detail={"status": "not_ready", "error": str(e)})
```

## 🔒 Security Considerations

### Input Validation and Sanitization

```python
# utils/security.py
import re
from typing import Any, List
import numpy as np

class SecurityValidator:
    def __init__(self):
        self.max_input_size = 10000  # Maximum input size in bytes
        self.allowed_characters = re.compile(r'^[a-zA-Z0-9\s\-_.,:;()[\]{}]*$')
    
    def validate_input_size(self, data: Any) -> bool:
        """Validate input size to prevent DoS attacks"""
        try:
            size = len(str(data).encode('utf-8'))
            return size <= self.max_input_size
        except:
            return False
    
    def sanitize_string(self, input_string: str) -> str:
        """Sanitize string input"""
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>&"\']', '', input_string)
        
        # Limit length
        max_length = 1000
        return sanitized[:max_length]
    
    def validate_numeric_array(self, array: List[float], expected_shape: tuple) -> bool:
        """Validate numeric array input"""
        try:
            arr = np.array(array)
            
            # Check for NaN or infinite values
            if np.isnan(arr).any() or np.isinf(arr).any():
                return False
            
            # Check shape
            if arr.shape != expected_shape:
                return False
            
            # Check for reasonable value ranges
            if np.abs(arr).max() > 1e6:  # Arbitrary large value threshold
                return False
            
            return True
        except:
            return False
    
    def check_sql_injection_patterns(self, input_string: str) -> bool:
        """Check for SQL injection patterns"""
        sql_patterns = [
            r'(\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b)',
            r'(\b(or|and)\b.*=.*)',
            r'(--|#|/\*|\*/)',
            r'(\b(xp_|sp_)\w+)',
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, input_string, re.IGNORECASE):
                return False
        
        return True

def validate_prediction_request(features: List[float]) -> bool:
    """Validate prediction request"""
    validator = SecurityValidator()
    
    # Validate input size
    if not validator.validate_input_size(features):
        return False
    
    # Validate numeric array
    if not validator.validate_numeric_array(features, (10,)):  # Example shape
        return False
    
    return True
```

### Rate Limiting

```python
# utils/rate_limiter.py
import time
from collections import defaultdict
from typing import Dict
from fastapi import HTTPException

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60, requests_per_hour: int = 1000):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.requests = defaultdict(list)  # client_id: [timestamps]
    
    def is_rate_limited(self, client_id: str) -> bool:
        """Check if client is rate limited"""
        now = time.time()
        client_requests = self.requests[client_id]
        
        # Remove old requests (older than 1 hour)
        self.requests[client_id] = [
            timestamp for timestamp in client_requests 
            if now - timestamp < 3600
        ]
        
        # Check hourly limit
        if len(self.requests[client_id]) >= self.requests_per_hour:
            return True
        
        # Check minute limit (requests in last 60 seconds)
        recent_requests = [
            timestamp for timestamp in self.requests[client_id] 
            if now - timestamp < 60
        ]
        
        if len(recent_requests) >= self.requests_per_minute:
            return True
        
        # Record this request
        self.requests[client_id].append(now)
        
        return False
    
    def get_rate_limit_info(self, client_id: str) -> Dict[str, int]:
        """Get rate limit information for client"""
        now = time.time()
        client_requests = self.requests.get(client_id, [])
        
        # Recent requests (last minute)
        recent_requests = [
            timestamp for timestamp in client_requests 
            if now - timestamp < 60
        ]
        
        # Hourly requests (last hour)
        hourly_requests = [
            timestamp for timestamp in client_requests 
            if now - timestamp < 3600
        ]
        
        return {
            "requests_per_minute_used": len(recent_requests),
            "requests_per_minute_limit": self.requests_per_minute,
            "requests_per_hour_used": len(hourly_requests),
            "requests_per_hour_limit": self.requests_per_hour,
            "remaining_minute": max(0, self.requests_per_minute - len(recent_requests)),
            "remaining_hour": max(0, self.requests_per_hour - len(hourly_requests))
        }

def rate_limit_middleware(rate_limiter: RateLimiter, client_id_header: str = "X-Client-ID"):
    """Rate limiting middleware for FastAPI"""
    async def middleware(request, call_next):
        client_id = request.headers.get(client_id_header, "anonymous")
        
        if rate_limiter.is_rate_limited(client_id):
            rate_info = rate_limiter.get_rate_limit_info(client_id)
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "rate_limit_info": rate_info
                }
            )
        
        response = await call_next(request)
        
        # Add rate limit headers
        rate_info = rate_limiter.get_rate_limit_info(client_id)
        response.headers["X-RateLimit-Remaining-Minute"] = str(rate_info["remaining_minute"])
        response.headers["X-RateLimit-Remaining-Hour"] = str(rate_info["remaining_hour"])
        
        return response
```

## 🚀 Deployment Strategies

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health/live')"

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-service
  labels:
    app: ml-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-service
  template:
    metadata:
      labels:
        app: ml-service
    spec:
      containers:
      - name: ml-service
        image: your-registry/ml-service:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/models"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        startupProbe:
          httpGet:
            path: /health/startup
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          failureThreshold: 30
        volumeMounts:
        - name: model-storage
          mountPath: /models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: ml-service
spec:
  selector:
    app: ml-service
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## 🔧 Troubleshooting Guide

### Common Issues and Solutions

#### 1. Model Loading Issues

**Problem**: Model fails to load on startup
```
Error: Failed to load model: File not found
```

**Solution**:
```python
# Check model file existence and permissions
import os
from pathlib import Path

def validate_model_files(model_path: str):
    """Validate model files before loading"""
    model_file = Path(model_path)
    
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not os.access(model_file, os.R_OK):
        raise PermissionError(f"Cannot read model file: {model_path}")
    
    # Check file size
    file_size = model_file.stat().st_size
    if file_size == 0:
        raise ValueError(f"Model file is empty: {model_path}")
    
    print(f"Model file validated: {model_path} ({file_size} bytes)")
    return True
```

#### 2. Memory Issues

**Problem**: High memory usage causing OOM errors
```
Error: Out of memory
```

**Solution**:
```python
# Memory-efficient model loading
import gc
import psutil

def monitor_memory():
    """Monitor memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
        "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        "percent": process.memory_percent()
    }

def load_model_memory_efficient(model_path: str):
    """Load model with memory monitoring"""
    # Check memory before loading
    memory_before = monitor_memory()
    print(f"Memory before loading: {memory_before['rss_mb']:.2f} MB")
    
    # Load model
    model = joblib.load(model_path)
    
    # Force garbage collection
    gc.collect()
    
    # Check memory after loading
    memory_after = monitor_memory()
    print(f"Memory after loading: {memory_after['rss_mb']:.2f} MB")
    print(f"Memory increase: {memory_after['rss_mb'] - memory_before['rss_mb']:.2f} MB")
    
    return model
```

#### 3. Performance Issues

**Problem**: Slow prediction response times
```
Response time > 1 second
```

**Solution**:
```python
# Performance profiling
import time
import cProfile
import pstats
from functools import wraps

def profile_performance(func):
    """Decorator for performance profiling"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        
        profiler.disable()
        
        # Print profiling stats
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(10)  # Top 10 functions
        
        print(f"Total execution time: {end_time - start_time:.4f} seconds")
        
        return result
    return wrapper

# Async optimization for I/O bound operations
async def optimized_batch_predict(model, features_batch):
    """Optimized batch prediction with async I/O"""
    # Use asyncio.gather for parallel processing
    tasks = []
    for features in features_batch:
        task = asyncio.create_task(predict_async(model, features))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results

async def predict_async(model, features):
    """Async prediction function"""
    # Run prediction in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, model.predict, features)
    return result
```

#### 4. Model Accuracy Issues

**Problem**: Model predictions are inaccurate

**Solution**:
```python
# Model monitoring and validation
def validate_model_performance(model, test_data, threshold=0.8):
    """Validate model performance against threshold"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    
    # Make predictions
    predictions = model.predict(test_data.features)
    
    # Calculate metrics
    accuracy = accuracy_score(test_data.targets, predictions)
    precision = precision_score(test_data.targets, predictions, average='weighted')
    recall = recall_score(test_data.targets, predictions, average='weighted')
    
    # Check against thresholds
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "threshold": threshold,
        "passed": accuracy >= threshold
    }
    
    if not metrics["passed"]:
        print(f"Model failed validation. Accuracy: {accuracy:.4f}, Threshold: {threshold}")
        # Trigger model retraining or alert
        trigger_model_retraining(metrics)
    
    return metrics

def trigger_model_retraining(metrics):
    """Trigger model retraining process"""
    # Log metrics for analysis
    logger.error("Model validation failed", extra={
        "accuracy": metrics["accuracy"],
        "threshold": metrics["threshold"],
        "action": "retraining_triggered"
    })
    
    # Queue retraining job
    # This would integrate with your job queue system
    print("Model retraining triggered due to performance degradation")
```

#### 5. Service Connection Issues

**Problem**: Cannot connect to ML service endpoints
```
Error: Connection refused
Error: Timeout connecting to service
```

**Solution**:
```python
# Connection troubleshooting
def diagnose_connection_issues(service_url: str, timeout: int = 10):
    """Diagnose connection issues with ML service"""
    import socket
    import urllib.parse
    
    try:
        # Parse URL
        parsed = urllib.parse.urlparse(service_url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == 'https' else 80)
        
        # Test basic connectivity
        print(f"Testing connection to {host}:{port}")
        
        # DNS resolution
        try:
            ip = socket.gethostbyname(host)
            print(f"DNS resolution successful: {host} -> {ip}")
        except socket.gaierror as e:
            print(f"DNS resolution failed: {e}")
            return False
        
        # Port connectivity
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                print(f"Port {port} is open")
                return True
            else:
                print(f"Port {port} is closed or blocked (error code: {result})")
                return False
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
            
    except Exception as e:
        print(f"URL parsing failed: {e}")
        return False

# Service health check with retry logic
def check_service_health_with_retry(service_url: str, max_retries: int = 3, retry_delay: int = 5):
    """Check service health with retry logic"""
    import requests
    from urllib3.util.retry import Retry
    from requests.adapters import HTTPAdapter
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        method_whitelist=["HEAD", "GET", "OPTIONS", "POST"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    try:
        # Test health endpoint
        health_url = f"{service_url.rstrip('/')}/health"
        print(f"Checking health endpoint: {health_url}")
        
        response = session.get(health_url, timeout=10)
        
        if response.status_code == 200:
            print(f"Service is healthy: {response.json()}")
            return True
        else:
            print(f"Service health check failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Service connection failed after {max_retries} retries: {e}")
        return False
```

#### 6. API Validation Errors

**Problem**: Request validation failures
```
Error: 422 Unprocessable Entity
Error: Validation error in request body
```

**Solution**:
```python
# Enhanced request validation
def validate_prediction_request(request_data: dict) -> dict:
    """Comprehensive request validation"""
    errors = []
    
    # Check required fields
    required_fields = ["features", "model_version"]
    for field in required_fields:
        if field not in request_data:
            errors.append(f"Missing required field: {field}")
    
    # Validate features
    if "features" in request_data:
        features = request_data["features"]
        
        # Check type
        if not isinstance(features, list):
            errors.append("Features must be a list")
        else:
            # Check length
            if len(features) == 0:
                errors.append("Features list cannot be empty")
            
            # Check for None values
            if any(f is None for f in features):
                errors.append("Features cannot contain None values")
            
            # Check for invalid numbers
            if any(not isinstance(f, (int, float)) for f in features):
                errors.append("All features must be numeric")
            
            # Check for NaN or infinity
            if any(str(f) in ['nan', 'inf', '-inf'] for f in features):
                errors.append("Features cannot contain NaN or infinity")
    
    # Validate model version
    if "model_version" in request_data:
        version = request_data["model_version"]
        if not isinstance(version, str) or not version.strip():
            errors.append("Model version must be a non-empty string")
    
    # Return validation result
    if errors:
        return {
            "valid": False,
            "errors": errors,
            "error_count": len(errors)
        }
    
    return {
        "valid": True,
        "features_count": len(request_data.get("features", [])),
        "model_version": request_data.get("model_version")
    }

# Example usage
def test_validation():
    """Test validation with various inputs"""
    test_cases = [
        {"features": [1.0, 2.0, 3.0], "model_version": "1.0.0"},  # Valid
        {"features": []},  # Empty features
        {"features": [1.0, None, 3.0]},  # None value
        {"model_version": "1.0.0"},  # Missing features
        {"features": "not a list"},  # Wrong type
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest case {i+1}: {test_case}")
        result = validate_prediction_request(test_case)
        print(f"Validation result: {result}")
```

#### 7. Dependency and Environment Issues

**Problem**: Missing dependencies or version conflicts
```
Error: ModuleNotFoundError: No module named 'xyz'
Error: Package version conflicts
```

**Solution**:
```python
# Environment validation and dependency checking
def validate_environment():
    """Validate that all required dependencies are available"""
    import importlib
    import sys
    
    required_packages = {
        'fastapi': '0.104.1',
        'uvicorn': '0.24.0',
        'pandas': '2.1.3',
        'scikit-learn': '1.3.2',
        'numpy': '1.24.3',
        'pydantic': '2.5.0'
    }
    
    missing_packages = []
    version_mismatches = []
    
    for package, required_version in required_packages.items():
        try:
            module = importlib.import_module(package)
            installed_version = getattr(module, '__version__', 'unknown')
            
            if installed_version != required_version:
                version_mismatches.append({
                    'package': package,
                    'required': required_version,
                    'installed': installed_version
                })
                
        except ImportError:
            missing_packages.append(package)
    
    # Report results
    if missing_packages:
        print(f"Missing packages: {missing_packages}")
        print(f"Install with: pip install {' '.join(missing_packages)}")
    
    if version_mismatches:
        print("Version mismatches found:")
        for mismatch in version_mismatches:
            print(f"  {mismatch['package']}: required {mismatch['required']}, installed {mismatch['installed']}")
    
    if not missing_packages and not version_mismatches:
        print("All dependencies are correctly installed")
        return True
    
    return False

def check_python_version():
    """Check Python version compatibility"""
    import sys
    
    current_version = sys.version_info
    min_version = (3, 8)
    max_version = (3, 12)
    
    if current_version < min_version:
        print(f"Python version too old: {current_version.major}.{current_version.minor}")
        print(f"Minimum required: {min_version[0]}.{min_version[1]}")
        return False
    
    if current_version >= max_version:
        print(f"Python version may not be fully supported: {current_version.major}.{current_version.minor}")
        print(f"Maximum tested: {max_version[0]}.{max_version[1]}")
        return False
    
    print(f"Python version compatible: {current_version.major}.{current_version.minor}")
    return True
```

#### 8. Model Inference Timeout Issues

**Problem**: Model inference takes too long or times out
```
Error: Request timeout after 30 seconds
Error: Model inference timeout
```

**Solution**:
```python
# Timeout handling and optimization
def handle_inference_timeout(model, features, timeout_seconds: int = 30):
    """Handle model inference with timeout protection"""
    import signal
    import threading
    
    class TimeoutException(Exception):
        pass
    
    def timeout_handler(signum, frame):
        raise TimeoutException("Model inference timed out")
    
    # Set timeout alarm (Unix only)
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
    
    try:
        start_time = time.time()
        
        # Perform inference
        prediction = model.predict(features)
        
        inference_time = time.time() - start_time
        print(f"Inference completed in {inference_time:.2f} seconds")
        
        return prediction, inference_time
        
    except TimeoutException as e:
        print(f"Inference timeout after {timeout_seconds} seconds")
        raise e
        
    finally:
        # Cancel timeout alarm
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)

def optimize_model_for_inference(model_path: str, optimization_level: str = "O1"):
    """Optimize model for faster inference"""
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    
    # Load original model
    model = joblib.load(model_path)
    
    if isinstance(model, RandomForestClassifier):
        # Reduce number of estimators for faster inference
        if optimization_level == "O2":
            model.n_estimators = max(10, model.n_estimators // 2)
            model.n_jobs = 1  # Avoid parallel overhead for single predictions
        
        # Reduce max depth if possible
        if hasattr(model, 'max_depth') and model.max_depth and model.max_depth > 10:
            model.max_depth = min(model.max_depth, 10)
    
    # Save optimized model
    optimized_path = model_path.replace('.pkl', '_optimized.pkl')
    joblib.dump(model, optimized_path)
    
    print(f"Model optimized and saved to {optimized_path}")
    return optimized_path
```

#### 9. Data Preprocessing Issues

**Problem**: Data preprocessing failures causing prediction errors
```
Error: Feature scaling failed
Error: Invalid data format for model input
```

**Solution**:
```python
# Data preprocessing validation and error handling
def validate_preprocessing_data(data: dict, expected_schema: dict) -> dict:
    """Validate data before preprocessing"""
    errors = []
    warnings = []
    
    # Check required fields against schema
    for field, field_info in expected_schema.items():
        if field_info.get('required', False) and field not in data:
            errors.append(f"Missing required field: {field}")
        
        if field in data:
            value = data[field]
            
            # Type validation
            expected_type = field_info.get('type')
            if expected_type and not isinstance(value, expected_type):
                errors.append(f"Field {field}: expected {expected_type.__name__}, got {type(value).__name__}")
            
            # Range validation
            if expected_type in [int, float]:
                min_val = field_info.get('min')
                max_val = field_info.get('max')
                
                if min_val is not None and value < min_val:
                    errors.append(f"Field {field}: value {value} below minimum {min_val}")
                
                if max_val is not None and value > max_val:
                    errors.append(f"Field {field}: value {value} above maximum {max_val}")
            
            # Missing data check
            if value is None:
                if field_info.get('allow_null', False):
                    warnings.append(f"Field {field}: contains null value")
                else:
                    errors.append(f"Field {field}: null value not allowed")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "error_count": len(errors),
        "warning_count": len(warnings)
    }

def safe_data_preprocessing(raw_data: dict, preprocessor) -> tuple:
    """Safe data preprocessing with error handling"""
    try:
        # Validate input data
        validation_result = validate_preprocessing_data(raw_data, get_expected_schema())
        
        if not validation_result["valid"]:
            raise ValueError(f"Data validation failed: {validation_result['errors']}")
        
        # Log warnings
        if validation_result["warnings"]:
            print(f"Data preprocessing warnings: {validation_result['warnings']}")
        
        # Perform preprocessing
        processed_data = preprocessor.transform(raw_data)
        
        # Validate output
        if processed_data is None:
            raise ValueError("Preprocessor returned None")
        
        if hasattr(processed_data, '__len__') and len(processed_data) == 0:
            raise ValueError("Preprocessor returned empty data")
        
        return processed_data, validation_result
        
    except Exception as e:
        print(f"Data preprocessing failed: {e}")
        raise e

def get_expected_schema() -> dict:
    """Get expected data schema for validation"""
    return {
        "feature1": {"type": float, "required": True, "min": 0.0, "max": 100.0},
        "feature2": {"type": float, "required": True, "allow_null": False},
        "feature3": {"type": int, "required": False, "min": 1},
        "category": {"type": str, "required": True}
    }
```

#### 10. Model Deployment Issues

**Problem**: Model deployment failures in production
```
Error: Model not found in production environment
Error: Model loading failed in container
```

**Solution**:
```python
# Deployment validation and health checks
def validate_deployment_environment():
    """Validate deployment environment"""
    import os
    import pathlib
    
    issues = []
    
    # Check required environment variables
    required_env_vars = [
        'MODEL_PATH',
        'SERVICE_PORT',
        'LOG_LEVEL',
        'MODEL_VERSION'
    ]
    
    for var in required_env_vars:
        if not os.environ.get(var):
            issues.append(f"Missing environment variable: {var}")
    
    # Check model file existence
    model_path = os.environ.get('MODEL_PATH')
    if model_path:
        model_file = pathlib.Path(model_path)
        if not model_file.exists():
            issues.append(f"Model file not found: {model_path}")
        elif not model_file.is_file():
            issues.append(f"Model path is not a file: {model_path}")
        elif model_file.stat().st_size == 0:
            issues.append(f"Model file is empty: {model_path}")
    
    # Check disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free / (1024**3)
        if free_gb < 1.0:  # Less than 1GB free
            issues.append(f"Low disk space: {free_gb:.2f} GB free")
    except Exception:
        pass
    
    # Check memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        if memory.percent > 95:
            issues.append(f"High memory usage: {memory.percent:.1f}%")
    except Exception:
        pass
    
    return {
        "deployment_ready": len(issues) == 0,
        "issues": issues,
        "issue_count": len(issues)
    }

def perform_health_check():
    """Comprehensive health check for deployment"""
    print("Performing deployment health check...")
    
    # Environment validation
    env_check = validate_deployment_environment()
    
    if not env_check["deployment_ready"]:
        print(f"❌ Environment issues found: {env_check['issues']}")
        return False
    
    # Model loading test
    try:
        model_path = os.environ.get('MODEL_PATH')
        if model_path:
            from utils.model_loader import ModelLoader
            loader = ModelLoader()
            model = loader.load_model(model_path)
            print("✅ Model loaded successfully")
        else:
            print("⚠️  No model path configured")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    # Service startup test
    try:
        import requests
        service_url = f"http://localhost:{os.environ.get('SERVICE_PORT', '8000')}"
        response = requests.get(f"{service_url}/health", timeout=5)
        
        if response.status_code == 200:
            print("✅ Service health check passed")
        else:
            print(f"❌ Service health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Service connection failed: {e}")
        return False
    
    print("✅ All deployment health checks passed")
    return True

# Run deployment validation
if __name__ == "__main__":
    deployment_ready = perform_health_check()
    if not deployment_ready:
        print("Deployment validation failed - check logs above")
        exit(1)
    else:
        print("System ready for deployment")
```

## 🔄 Model Versioning

### Version Control Strategy

```python
# utils/model_lifecycle.py
from typing import Dict, Any, List
from datetime import datetime, timedelta
import json
from pathlib import Path

class ModelLifecycleManager:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.metadata_file = self.models_dir / "model_metadata.json"
        
    def register_model_version(
        self, 
        model_name: str, 
        version: str, 
        metrics: Dict[str, float],
        model_path: str,
        stage: str = "development"
    ):
        """Register a new model version"""
        metadata = self._load_metadata()
        
        model_info = {
            "name": model_name,
            "version": version,
            "metrics": metrics,
            "model_path": model_path,
            "stage": stage,
            "created_at": datetime.utcnow().isoformat(),
            "deployed_at": None
        }
        
        if model_name not in metadata:
            metadata[model_name] = []
        
        metadata[model_name].append(model_info)
        self._save_metadata(metadata)
        
        return model_info
    
    def promote_model(self, model_name: str, version: str, new_stage: str):
        """Promote model to new stage"""
        metadata = self._load_metadata()
        
        if model_name not in metadata:
            raise ValueError(f"Model {model_name} not found")
        
        for model_info in metadata[model_name]:
            if model_info["version"] == version:
                model_info["stage"] = new_stage
                if new_stage == "production":
                    model_info["deployed_at"] = datetime.utcnow().isoformat()
                break
        
        self._save_metadata(metadata)
    
    def get_model_info(self, model_name: str, version: str = None) -> Dict[str, Any]:
        """Get model information"""
        metadata = self._load_metadata()
        
        if model_name not in metadata:
            return None
        
        if version:
            for model_info in metadata[model_name]:
                if model_info["version"] == version:
                    return model_info
            return None
        else:
            # Return latest version
            return max(metadata[model_name], key=lambda x: x["created_at"])
    
    def list_model_versions(self, model_name: str, stage: str = None) -> List[Dict[str, Any]]:
        """List all versions of a model"""
        metadata = self._load_metadata()
        
        if model_name not in metadata:
            return []
        
        versions = metadata[model_name]
        
        if stage:
            versions = [v for v in versions if v["stage"] == stage]
        
        return sorted(versions, key=lambda x: x["created_at"], reverse=True)
    
    def rollback_model(self, model_name: str, target_version: str):
        """Rollback to previous model version"""
        # Get current production version
        current_prod = self.get_model_info(model_name, stage="production")
        
        if not current_prod:
            raise ValueError("No production model found")
        
        # Demote current production
        self.promote_model(model_name, current_prod["version"], "rollback")
        
        # Promote target version to production
        self.promote_model(model_name, target_version, "production")
        
        return {
            "previous_version": current_prod["version"],
            "new_version": target_version,
            "rollback_time": datetime.utcnow().isoformat()
        }
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load model metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self, metadata: Dict[str, Any]):
        """Save model metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
```

## 📚 References & Resources

### Machine Learning Frameworks
- **Scikit-learn**: https://scikit-learn.org/
- **TensorFlow**: https://www.tensorflow.org/
- **PyTorch**: https://pytorch.org/
- **ONNX**: https://onnx.ai/
- **MLflow**: https://mlflow.org/

### API Development
- **FastAPI**: https://fastapi.tiangolo.com/
- **Pydantic**: https://pydantic-docs.helpmanual.io/
- **OpenAPI**: https://swagger.io/specification/
- **AsyncIO**: https://docs.python.org/3/library/asyncio.html

### Deployment & Operations
- **Docker**: https://docs.docker.com/
- **Kubernetes**: https://kubernetes.io/docs/
- **Prometheus**: https://prometheus.io/docs/
- **Grafana**: https://grafana.com/docs/

### Monitoring & Observability
- **Prometheus Client**: https://github.com/prometheus/client_python
- **Structured Logging**: https://docs.python.org/3/howto/logging.html
- **Distributed Tracing**: https://opentelemetry.io/docs/

### Security
- **OWASP API Security**: https://owasp.org/www-project-api-security/
- **Python Security**: https://python-security.readthedocs.io/
- **Input Validation**: https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html

### Research Papers
- "Machine Learning: The High-Interest Credit Card of Technical Debt" - Google Research
- "Hidden Technical Debt in Machine Learning Systems" - NeurIPS 2015
- "The ML Test Score: A Rubric for ML Production Readiness" - Google Research
- "Rules of Machine Learning: Best Practices for ML Engineering" - Google Research

---

**📖 This guide is continuously updated as the project evolves. For the latest information, please refer to the project's GitHub repository.**