# Lead Scoring System

This directory contains the Lead Scoring System - an ML-powered system that predicts the likelihood of leads converting to customers using classification algorithms and behavioral analysis.

## Features

- **Lead Scoring Models**: XGBoost and Random Forest classifiers for lead conversion prediction
- **Behavioral Analysis**: Track and analyze lead interactions and engagement
- **Feature Engineering**: Automated feature extraction from lead data
- **Model Comparison**: Compare performance of different algorithms
- **Threshold Optimization**: Find optimal conversion thresholds
- **Real-time Scoring**: Score new leads in real-time
- **Performance Monitoring**: Track model performance and accuracy

## API Endpoints

- `POST /score` - Score a single lead
- `POST /score/batch` - Score multiple leads
- `POST /train` - Train models on new data
- `GET /model/info` - Get model information and performance
- `GET /model/performance` - Get detailed model performance metrics
- `POST /generate-sample-data` - Generate sample lead data for testing
- `GET /health` - Health check endpoint

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Start the API Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8004
```

### Example Usage

#### Score a Single Lead
```python
import requests

lead_data = {
    "lead_id": "LEAD001",
    "company_size": "medium",
    "industry": "technology",
    "job_title": "manager",
    "lead_source": "website",
    "engagement_score": 75,
    "website_visits": 5,
    "email_opens": 3,
    "form_submissions": 2,
    "demo_requests": 1,
    "content_downloads": 4,
    "social_media_engagement": 2,
    "days_since_last_activity": 2,
    "budget_range": "medium",
    "authority_level": "influencer",
    "timeline": "3_months",
    "pain_points": ["efficiency", "cost"],
    "competitor_usage": False,
    "marketing_qualified": True,
    "sales_qualified": False
}

response = requests.post("http://localhost:8004/score", json=lead_data)
result = response.json()
print(f"Lead Score: {result['score']}")
print(f"Conversion Probability: {result['conversion_probability']:.2%}")
print(f"Recommendation: {result['recommendation']}")
```

## Model Details

### Features Used
- **Demographic**: Company size, industry, job title
- **Behavioral**: Website visits, email engagement, content downloads
- **Qualification**: Budget, authority, timeline, pain points
- **Engagement**: Demo requests, form submissions, social media activity

### Algorithms
- **XGBoost**: Gradient boosting for high accuracy
- **Random Forest**: Ensemble method for robust predictions
- **Logistic Regression**: Baseline model for comparison

### Performance Metrics
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positive rate for conversions
- **Recall**: Ability to identify actual conversions
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve

## Testing

Run the API tests:
```bash
python test_api.py
```

## Model Training

For advanced model training:
```bash
cd notebooks
python train_lead_scoring_models.py
```

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8004/docs`
- ReDoc: `http://localhost:8004/redoc`