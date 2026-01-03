# 📊 Customer Churn Prediction - Comprehensive Documentation

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
The Customer Churn Prediction service is an advanced machine learning microservice that predicts the likelihood of customers leaving a service (churning). It uses XGBoost for predictions and SHAP (SHapley Additive exPlanations) for model interpretability, providing actionable insights for customer retention strategies.

### Business Problem
- **Problem**: Customer acquisition costs 5-25x more than retention. Companies lose 10-30% of customers annually, leading to significant revenue loss.
- **Solution**: Proactive identification of at-risk customers using predictive analytics, enabling targeted retention campaigns.
- **Impact**: 
  - Reduce churn rate by 15-25%
  - Increase customer lifetime value by 20-40%
  - Optimize marketing spend by targeting high-risk customers
  - Improve customer satisfaction through proactive engagement

### Expected Outcomes
- **Primary Metrics**:
  - Prediction Accuracy: >85%
  - AUC-ROC Score: >0.90
  - Precision (Churn): >80%
  - Recall (Churn): >75%
  - Response Time: <200ms per prediction
  - API Uptime: >99.5%

- **Success Criteria**:
  - Identify 80%+ of customers who will churn
  - Provide explainable predictions with key risk factors
  - Enable batch processing of 10,000+ customers
  - Support real-time prediction in customer service workflows
  - Reduce false positives to minimize unnecessary interventions

---

## Technical Implementation

### Technology Stack

#### Core Technologies
1. **FastAPI** (v0.104.1)
   - High-performance async web framework
   - Automatic OpenAPI documentation
   - Built-in validation with Pydantic
   - WebSocket support for real-time updates

2. **XGBoost** (v1.7.6)
   - Gradient boosting decision trees
   - Handles missing values automatically
   - Built-in regularization (L1/L2)
   - Feature importance extraction
   - Superior performance on structured data

3. **SHAP** (v0.42.1)
   - Model-agnostic explainability
   - Individual prediction explanations
   - Feature importance visualization
   - Additive feature attribution

4. **Scikit-learn** (v1.3.2)
   - Data preprocessing pipelines
   - Train/test splitting
   - Cross-validation
   - Evaluation metrics

5. **Pandas** (v2.0.3)
   - Data manipulation and analysis
   - CSV/JSON processing
   - Feature engineering

#### Supporting Libraries
- **NumPy**: Numerical computations
- **Joblib**: Model serialization
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation and settings

### Technology Alternatives

| Component | Current Choice | Alternative 1 | Alternative 2 | Why Current Choice |
|-----------|---------------|---------------|---------------|-------------------|
| **ML Algorithm** | XGBoost | LightGBM | CatBoost | Best accuracy, feature importance, handles imbalanced data |
| | | Random Forest | Neural Network | Faster training, interpretable, production-ready |
| **Explainability** | SHAP | LIME | ELI5 | Theoretically sound, consistent, handles any model |
| **Web Framework** | FastAPI | Flask | Django REST | Async support, auto docs, type safety, performance |
| **Validation** | Pydantic | Marshmallow | Cerberus | Native FastAPI integration, type hints, performance |
| **Serialization** | Joblib | Pickle | ONNX | Scikit-learn compatibility, compression, versioning |

### Why XGBoost?
1. **High Accuracy**: State-of-the-art performance on tabular data
2. **Speed**: Optimized C++ implementation with parallelization
3. **Robustness**: Handles missing values, outliers, and imbalanced data
4. **Regularization**: Built-in L1/L2 to prevent overfitting
5. **Feature Engineering**: Automatic feature interaction discovery
6. **Interpretability**: Native feature importance + SHAP integration
7. **Production Ready**: Stable, well-maintained, widely adopted

### Why SHAP?
1. **Theoretical Foundation**: Based on game theory (Shapley values)
2. **Consistency**: Same feature contribution regardless of other features
3. **Local Explanations**: Explains individual predictions
4. **Global Insights**: Aggregate feature importance across dataset
5. **Model Agnostic**: Works with any ML model
6. **Actionable**: Provides specific factors driving each prediction

### Code Function Explanations

#### 1. Model Training (`model.py`)

```python
class ChurnPredictionModel:
    """
    Comprehensive churn prediction model with SHAP explainability
    """
    
    def __init__(self):
        """
        Initialize model components:
        - model: XGBoost classifier
        - preprocessor: Feature engineering pipeline
        - shap_explainer: SHAP TreeExplainer for interpretability
        - model_info: Metadata about training performance
        """
    
    def train_model(self, algorithm='xgboost'):
        """
        Train churn prediction model
        
        Process:
        1. Load and validate training data
        2. Handle missing values and outliers
        3. Encode categorical variables (one-hot encoding)
        4. Scale numerical features (standardization)
        5. Handle class imbalance (SMOTE or class weights)
        6. Train XGBoost with hyperparameter tuning
        7. Validate with 5-fold cross-validation
        8. Initialize SHAP explainer
        9. Calculate and store performance metrics
        
        Hyperparameters:
        - max_depth: 6 (tree depth)
        - learning_rate: 0.1 (step size shrinkage)
        - n_estimators: 200 (number of trees)
        - subsample: 0.8 (row sampling)
        - colsample_bytree: 0.8 (column sampling)
        - scale_pos_weight: ratio of negative/positive (handles imbalance)
        
        Returns: Dict with accuracy, precision, recall, F1, AUC
        """
    
    def preprocess_data(self, data):
        """
        Feature engineering pipeline
        
        Steps:
        1. Handle missing values:
           - Numerical: Median imputation
           - Categorical: Mode imputation or 'Unknown'
        
        2. Feature encoding:
           - One-hot encoding for nominal categories
           - Label encoding for ordinal categories
           - Binary encoding for yes/no fields
        
        3. Feature scaling:
           - StandardScaler for numerical features
           - Preserves distribution shape
        
        4. Feature creation:
           - Tenure groups (new/mid/long-term customers)
           - Charge ratios (monthly/total)
           - Service combination features
           - Contract type indicators
        
        5. Feature selection:
           - Remove highly correlated features (>0.95)
           - Drop low-variance features
        
        Returns: Preprocessed feature matrix
        """
    
    def predict(self, customer_data, include_shap=False):
        """
        Make churn prediction for single customer
        
        Process:
        1. Validate input data schema
        2. Preprocess features (same pipeline as training)
        3. Generate prediction probability
        4. Apply threshold (default: 0.5)
        5. Calculate risk score (0-100)
        6. Determine confidence level (Low/Medium/High)
        7. Extract top risk factors
        8. Optional: Generate SHAP explanation
        
        Risk Score Calculation:
        - 0-30: Low risk (retain naturally)
        - 31-60: Medium risk (monitor)
        - 61-100: High risk (immediate intervention)
        
        Confidence Levels:
        - High: probability > 0.8 or < 0.2
        - Medium: probability 0.6-0.8 or 0.2-0.4
        - Low: probability 0.4-0.6 (uncertain)
        
        Returns: Prediction dict with probability, risk factors, SHAP values
        """
    
    def get_shap_values(self, customer_data):
        """
        Generate SHAP explanations for prediction
        
        SHAP Values Interpretation:
        - Positive value: Feature increases churn probability
        - Negative value: Feature decreases churn probability
        - Magnitude: Strength of feature's impact
        
        Example:
        {
            "tenure": -0.15,           # Long tenure reduces churn
            "MonthlyCharges": 0.25,    # High charges increase churn
            "Contract": -0.30,         # Long contract reduces churn
            "TechSupport": -0.10       # Having tech support reduces churn
        }
        
        Returns: Dict of feature names to SHAP values
        """
    
    def get_feature_importance(self):
        """
        Extract global feature importance from model
        
        Methods:
        1. Gain: Average training loss reduction from splits on feature
        2. Weight: Number of times feature is used in splits
        3. Cover: Average coverage (samples affected) by feature
        
        Returns: Sorted dict of features and importance scores
        """
```

#### 2. API Endpoints (`main.py`)

```python
@app.post("/predict", response_model=ChurnPrediction)
async def predict_churn(customer_data: CustomerData, include_shap: bool = False):
    """
    Predict churn for single customer
    
    Request Body:
    {
        "customer_id": "C123",
        "gender": "Male",
        "senior_citizen": 0,
        "partner": "Yes",
        "dependents": "No",
        "tenure": 24,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "Yes",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 89.99,
        "total_charges": 2159.76
    }
    
    Response:
    {
        "customer_id": "C123",
        "churn_probability": 0.78,
        "churn_prediction": "Yes",
        "confidence": "High",
        "risk_score": 78.0,
        "key_factors": [
            "Month-to-month contract (+0.35)",
            "High monthly charges (+0.25)",
            "No tech support (+0.18)"
        ],
        "shap_values": {...},  // If include_shap=true
        "recommendation": "High priority retention"
    }
    
    Use Cases:
    - Real-time customer service dashboard
    - Trigger retention workflow
    - Risk assessment in CRM
    """

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_churn(batch_request: BatchPredictionRequest):
    """
    Batch prediction for multiple customers
    
    Features:
    - Process up to 10,000 customers per request
    - Parallel processing for speed
    - Summary statistics (churn rate, avg probability)
    - Error handling per customer (continues on failure)
    
    Use Cases:
    - Daily batch scoring of customer base
    - Campaign target list generation
    - Quarterly risk assessment reports
    - A/B testing different models
    
    Response includes:
    - Individual predictions for each customer
    - Aggregate statistics (total churn, average risk)
    - Processing time and throughput metrics
    """

@app.post("/train")
async def train_model():
    """
    Retrain model with latest data
    
    Process:
    1. Load training data (CSV or database)
    2. Validate data quality
    3. Train new model
    4. Evaluate on test set
    5. Compare with existing model
    6. Save if performance improves
    7. Update SHAP explainer
    
    Triggers:
    - Manual: API call or admin dashboard
    - Scheduled: Cron job (weekly/monthly)
    - Automated: Performance degradation detected
    
    Returns: Training metrics and model info
    """

@app.get("/model/feature_importance")
async def get_feature_importance():
    """
    Get global feature importance
    
    Use Cases:
    - Understand key churn drivers across all customers
    - Prioritize business interventions
    - Report to stakeholders
    - Feature engineering guidance
    
    Returns: Ranked list of features with importance scores
    """

@app.get("/analytics/summary")
async def get_analytics_summary():
    """
    Generate analytics summary
    
    Metrics:
    - Overall churn rate
    - Average customer tenure
    - Revenue metrics (MRR, ARPU)
    - Contract distribution
    - Service adoption rates
    - Demographic breakdowns
    
    Use Cases:
    - Executive dashboards
    - Business intelligence reports
    - Trend analysis
    """
```

#### 3. Data Validation (`schemas.py`)

```python
class CustomerData(BaseModel):
    """
    Pydantic model for customer input validation
    
    Features:
    - Automatic type checking
    - Range validation
    - Enum validation for categorical fields
    - Custom validators for business rules
    - Clear error messages
    
    Benefits:
    - Prevents invalid data from reaching model
    - Documents API contract
    - Reduces debugging time
    - Improves data quality
    """
    
    customer_id: str
    gender: str  # "Male", "Female"
    senior_citizen: int  # 0 or 1
    partner: str  # "Yes", "No"
    dependents: str  # "Yes", "No"
    tenure: int  # Range: 0-72 months
    
    # Services
    phone_service: str
    multiple_lines: str  # "Yes", "No", "No phone service"
    internet_service: str  # "DSL", "Fiber optic", "No"
    
    # Add-ons
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    
    # Contract
    contract: str  # "Month-to-month", "One year", "Two year"
    paperless_billing: str  # "Yes", "No"
    payment_method: str  # "Electronic check", "Mailed check", etc.
    
    # Charges
    monthly_charges: float  # Range: 0-200
    total_charges: float  # Range: 0-10000
    
    @validator('tenure')
    def validate_tenure(cls, v):
        if v < 0 or v > 72:
            raise ValueError('Tenure must be between 0 and 72 months')
        return v
    
    @validator('monthly_charges')
    def validate_monthly_charges(cls, v):
        if v < 0 or v > 200:
            raise ValueError('Monthly charges must be between 0 and 200')
        return v
```

---

## Architectural Documentation

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     CLIENT APPLICATIONS                          │
│   (CRM Systems, Admin Dashboards, Mobile Apps, Batch Jobs)      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ HTTPS REST API
                            │ JSON Payloads
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                      FASTAPI APPLICATION                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │           API Layer (main.py)                              │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐ │ │
│  │  │  /predict    │  │  /predict/   │  │  /train         │ │ │
│  │  │  (Single)    │  │  batch       │  │  (Retrain)      │ │ │
│  │  └──────────────┘  └──────────────┘  └─────────────────┘ │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐ │ │
│  │  │  /model/     │  │  /analytics/ │  │  /health        │ │ │
│  │  │  info        │  │  summary     │  │  (Status)       │ │ │
│  │  └──────────────┘  └──────────────┘  └─────────────────┘ │ │
│  └────────────────────────────────────────────────────────────┘ │
│                            │                                     │
│  ┌────────────────────────▼────────────────────────────────────┐│
│  │         Validation Layer (schemas.py)                       ││
│  │  - Pydantic models for request/response                     ││
│  │  - Type checking and data validation                        ││
│  │  - Business rule enforcement                                ││
│  └────────────────────────┬────────────────────────────────────┘│
│                            │                                     │
│  ┌────────────────────────▼────────────────────────────────────┐│
│  │       Business Logic Layer (model.py)                       ││
│  │                                                              ││
│  │  ┌───────────────────────────────────────────────────────┐ ││
│  │  │  ChurnPredictionModel Class                           │ ││
│  │  │  ├── Data Preprocessing Pipeline                      │ ││
│  │  │  │   - Missing value imputation                       │ ││
│  │  │  │   - Feature encoding (one-hot, label)              │ ││
│  │  │  │   - Feature scaling (standardization)              │ ││
│  │  │  │   - Feature engineering                            │ ││
│  │  │  │                                                     │ ││
│  │  │  ├── XGBoost Classifier                               │ ││
│  │  │  │   - Gradient boosted trees                         │ ││
│  │  │  │   - Hyperparameter optimization                    │ ││
│  │  │  │   - Cross-validation                               │ ││
│  │  │  │                                                     │ ││
│  │  │  └── SHAP Explainer                                   │ ││
│  │  │      - TreeExplainer for XGBoost                      │ ││
│  │  │      - Individual prediction explanations             │ ││
│  │  │      - Feature importance aggregation                 │ ││
│  │  └───────────────────────────────────────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────┘│
│                            │                                     │
│  ┌────────────────────────▼────────────────────────────────────┐│
│  │         Model Storage (In-Memory + Disk)                    ││
│  │  - Trained XGBoost model (serialized)                       ││
│  │  - SHAP explainer (cached)                                  ││
│  │  - Model metadata and metrics                               ││
│  │  - Feature preprocessors                                    ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ Persist/Load
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    PERSISTENT STORAGE                            │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │  Model Files     │  │  Training Data   │  │  Logs         │ │
│  │  - .joblib       │  │  - CSV/Database  │  │  - Predictions│ │
│  │  - metadata.json │  │  - Validation    │  │  - Errors     │ │
│  └──────────────────┘  └──────────────────┘  └───────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
┌──────────────┐
│ Customer     │
│ Features     │
│ (19 fields)  │
└──────┬───────┘
       │
       ▼
┌──────────────────────────┐
│ Data Validation          │
│ - Type checking          │
│ - Range validation       │
│ - Business rules         │
└──────┬───────────────────┘
       │ Valid ✓
       ▼
┌──────────────────────────┐
│ Feature Preprocessing    │
│ - Handle missing values  │
│ - Encode categoricals    │
│ - Scale numericals       │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│ Feature Engineering      │
│ - Tenure groups          │
│ - Service combinations   │
│ - Charge ratios          │
└──────┬───────────────────┘
       │ Feature Matrix
       │ (45 features)
       ▼
┌──────────────────────────┐
│ XGBoost Model            │
│ - 200 decision trees     │
│ - Gradient boosting      │
│ - Class balancing        │
└──────┬───────────────────┘
       │
       ├─────────────────────────────┐
       │                             │
       ▼                             ▼
┌──────────────────┐      ┌─────────────────────┐
│ Prediction       │      │ SHAP Explanation    │
│ - Probability    │      │ - Feature impacts   │
│ - Binary class   │      │ - Contribution      │
└──────┬───────────┘      └─────────┬───────────┘
       │                            │
       └──────────┬─────────────────┘
                  │
                  ▼
┌────────────────────────────────────┐
│ Risk Assessment                    │
│ - Calculate risk score (0-100)     │
│ - Determine confidence level       │
│ - Identify top risk factors        │
│ - Generate recommendation          │
└──────┬─────────────────────────────┘
       │
       ▼
┌────────────────────────────────────┐
│ JSON Response                      │
│ {                                  │
│   "churn_probability": 0.78,       │
│   "prediction": "Yes",             │
│   "risk_score": 78,                │
│   "key_factors": [...],            │
│   "recommendation": "..."          │
│ }                                  │
└────────────────────────────────────┘
```

### Execution Flowchart - Single Prediction

```
START
  │
  ▼
[FastAPI App Initialization]
  │
  ▼
[Load Trained Model & SHAP Explainer]
  │
  ├─ Success? ─No──► [Log Error] ─► [503 Service Unavailable]
  │                                          │
  Yes                                       END
  │
  ▼
[Listen for HTTP Requests]
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
[Extract Customer Features]
  │
  ▼
[Check Missing Values]
  │
  ├─ Has Missing? ─Yes──► [Impute with Median/Mode]
  │                                │
  No                              │
  │◄──────────────────────────────┘
  │
  ▼
[Encode Categorical Features]
  │ (One-hot encoding)
  │
  ▼
[Scale Numerical Features]
  │ (Standardization: μ=0, σ=1)
  │
  ▼
[Engineer Additional Features]
  │ (Tenure groups, charge ratios)
  │
  ▼
[Create Feature Matrix]
  │ (45 features)
  │
  ▼
[XGBoost Prediction]
  │ (200 trees vote)
  │
  ▼
[Get Probability Scores]
  │ P(Churn=0), P(Churn=1)
  │
  ▼
[Apply Threshold (0.5)]
  │
  ├─ P(Churn) > 0.5? ─Yes──► [Prediction: CHURN]
  │                                │
  No                              │
  │                                │
  └► [Prediction: RETAIN]         │
         │                         │
         └────────┬────────────────┘
                  │
                  ▼
        [Calculate Risk Score]
         │ risk_score = probability * 100
         │
         ▼
        [Determine Confidence Level]
         │ High: prob > 0.8 or < 0.2
         │ Medium: prob 0.6-0.8 or 0.2-0.4
         │ Low: prob 0.4-0.6
         │
         ▼
        [Include SHAP?]
         │
         ├─ Yes ──► [Calculate SHAP Values]
         │              │ For each feature
         │              │
         │              ▼
         │          [Get Feature Contributions]
         │              │
         │              ▼
         │          [Sort by Absolute Value]
         │              │
         No             │
         │              │
         └─────┬────────┘
               │
               ▼
        [Extract Top 5 Risk Factors]
               │
               ▼
        [Generate Recommendation]
         │ High risk: "Immediate intervention"
         │ Medium: "Monitor and engage"
         │ Low: "Standard service"
               │
               ▼
        [Format JSON Response]
               │
               ▼
        [Return 200 OK]
               │
               ▼
              END
```

### Machine Learning Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                            │
└─────────────────────────────────────────────────────────────────┘

Step 1: Data Collection
├── Load training data (CSV/Database)
├── Initial data validation
└── Verify data quality (5000+ samples required)
     │
     ▼
Step 2: Exploratory Data Analysis
├── Check class distribution (churn vs non-churn)
├── Identify missing values
├── Detect outliers
├── Analyze feature correlations
└── Visualize distributions
     │
     ▼
Step 3: Data Preprocessing
├── Handle missing values
│   ├── TotalCharges: Convert to numeric, fill with 0
│   ├── Numerical: Median imputation
│   └── Categorical: Mode imputation
├── Remove duplicates
├── Drop irrelevant columns (customerID)
└── Validate data types
     │
     ▼
Step 4: Feature Engineering
├── Create tenure groups (0-12, 13-24, 25-48, 49-72)
├── Calculate charge ratios (monthly/total)
├── Service combination features
├── Contract type indicators
├── Payment method grouping
└── Senior citizen interactions
     │
     ▼
Step 5: Feature Encoding
├── One-hot encode nominal categories
│   ├── Gender, InternetService, Contract
│   └── PaymentMethod, MultipleLines
├── Label encode binary categories
│   └── Partner, Dependents, PhoneService
└── Keep numerical as-is (tenure, charges)
     │
     ▼
Step 6: Feature Scaling
├── StandardScaler for numerical features
│   ├── Mean = 0, Std Dev = 1
│   └── Preserves distribution shape
└── No scaling needed for encoded features
     │
     ▼
Step 7: Handle Class Imbalance
├── Calculate class ratio (typically 3:1)
├── Option 1: SMOTE (Synthetic Minority Oversampling)
├── Option 2: Class weights in XGBoost
└── Validate balanced distribution
     │
     ▼
Step 8: Train/Test Split
├── Split ratio: 80% train, 20% test
├── Stratified sampling (preserve class ratio)
└── Set random seed for reproducibility
     │
     ▼
Step 9: Hyperparameter Tuning
├── Grid search or random search
├── Parameters:
│   ├── max_depth: [3, 5, 7]
│   ├── learning_rate: [0.01, 0.1, 0.3]
│   ├── n_estimators: [100, 200, 300]
│   ├── subsample: [0.7, 0.8, 0.9]
│   └── colsample_bytree: [0.7, 0.8, 0.9]
├── 5-fold cross-validation
└── Select best parameters
     │
     ▼
Step 10: Model Training
├── Train XGBoost with best parameters
├── Monitor training progress
├── Early stopping (patience=10)
└── Save training history
     │
     ▼
Step 11: Model Evaluation
├── Predictions on test set
├── Calculate metrics:
│   ├── Accuracy
│   ├── Precision, Recall, F1-Score
│   ├── AUC-ROC
│   ├── Confusion Matrix
│   └── Classification Report
├── Feature importance analysis
└── Validate performance > threshold
     │
     ▼
Step 12: SHAP Initialization
├── Create TreeExplainer
├── Calculate SHAP values on sample
├── Verify explanation quality
└── Cache explainer
     │
     ▼
Step 13: Model Serialization
├── Save XGBoost model (.joblib)
├── Save preprocessors (scalers, encoders)
├── Save metadata (features, version, metrics)
└── Version control (timestamp)
     │
     ▼
Step 14: Model Validation
├── Load saved model
├── Test predictions on validation set
├── Compare with training metrics
└── Approve for deployment
     │
     ▼
   END

┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE                            │
└─────────────────────────────────────────────────────────────────┘

Step 1: Receive Request
     │
     ▼
Step 2: Validate Input (Pydantic)
     │
     ▼
Step 3: Preprocess (Same as Training)
     │
     ▼
Step 4: Predict
     │
     ▼
Step 5: Explain (SHAP)
     │
     ▼
Step 6: Format Response
     │
     ▼
   END