# 📊 Visual Diagrams Guide - AI/ML Projects

This document contains visual diagrams for understanding the architecture, data flow, and relationships across all AI/ML projects. These diagrams can be rendered using Mermaid or viewed as ASCII art.

## Table of Contents
1. [Project Relationship Diagrams](#project-relationship-diagrams)
2. [Architecture Diagrams](#architecture-diagrams)
3. [Data Flow Diagrams](#data-flow-diagrams)
4. [Learning Path Diagrams](#learning-path-diagrams)
5. [Technology Stack Diagrams](#technology-stack-diagrams)

---

## Project Relationship Diagrams

### Project Classification by Domain (Mermaid)

```mermaid
graph TB
    Root[AI/ML Projects - 36 Total]
    
    Root --> NLP[NLP Projects - 8]
    Root --> CV[Computer Vision - 5]
    Root --> Pred[Predictive Analytics - 11]
    Root --> Rec[Recommendation Systems - 4]
    Root --> Opt[Optimization - 3]
    Root --> Speech[Speech Processing - 1]
    Root --> MLOps[MLOps - 1]
    Root --> Other[Other - 3]
    
    NLP --> NLP1[spam-classifier]
    NLP --> NLP2[sentiment-service]
    NLP --> NLP3[fake-news-detector]
    NLP --> NLP4[resume-analyzer]
    NLP --> NLP5[chatbot-api]
    NLP --> NLP6[summarization-api]
    NLP --> NLP7[text-to-sql]
    NLP --> NLP8[news-aggregator]
    
    CV --> CV1[digit-recognition]
    CV --> CV2[image-classification]
    CV --> CV3[face-recognition]
    CV --> CV4[quality-control-cv]
    CV --> CV5[image-classification-products]
    
    Pred --> P1[house-price-api]
    Pred --> P2[churn-prediction]
    Pred --> P3[credit-card-fraud]
    Pred --> P4[lead-scoring]
    Pred --> P5[sales-forecasting]
    Pred --> P6[loan-eligibility]
    Pred --> P7[stock-price-classifier]
    Pred --> P8[predictive-maintenance]
    Pred --> P9[clv-predictor]
    Pred --> P10[demand-forecasting-neural]
    Pred --> P11[energy-consumption-forecasting]
    
    Rec --> R1[movie-recommender]
    Rec --> R2[product-recommender]
    Rec --> R3[recommendation-system-collaborative]
    Rec --> R4[market-basket-analysis]
    
    Opt --> O1[price-optimization-engine]
    Opt --> O2[supply-chain-optimization]
    Opt --> O3[inventory-optimization]
    
    Speech --> S1[speech-to-text]
    
    MLOps --> M1[auto-retraining]
    
    Other --> OT1[customer-segmentation]
    Other --> OT2[product-demand-forecasting]
    Other --> OT3[Math]
    
    style Root fill:#f9f,stroke:#333,stroke-width:4px
    style NLP fill:#bbf,stroke:#333,stroke-width:2px
    style CV fill:#bfb,stroke:#333,stroke-width:2px
    style Pred fill:#fbb,stroke:#333,stroke-width:2px
    style Rec fill:#fbf,stroke:#333,stroke-width:2px
```

### Learning Difficulty Progression (Mermaid)

```mermaid
graph LR
    B1[Beginner] --> I1[Intermediate]
    I1 --> A1[Advanced]
    
    B1 --> B_Easy[spam-classifier<br/>house-price-api<br/>sentiment-service<br/>loan-eligibility]
    
    I1 --> I_Med[churn-prediction<br/>image-classification<br/>movie-recommender<br/>credit-card-fraud]
    
    A1 --> A_Hard[chatbot-api<br/>face-recognition<br/>text-to-sql<br/>auto-retraining]
    
    style B1 fill:#9f9,stroke:#333,stroke-width:2px
    style I1 fill:#ff9,stroke:#333,stroke-width:2px
    style A1 fill:#f99,stroke:#333,stroke-width:2px
```

### Technology Dependency Graph (Mermaid)

```mermaid
graph TB
    Python[Python 3.8+]
    
    Python --> FastAPI[FastAPI]
    Python --> ML[ML Libraries]
    Python --> Data[Data Libraries]
    
    FastAPI --> Uvicorn[Uvicorn]
    FastAPI --> Pydantic[Pydantic]
    
    ML --> SKL[Scikit-learn]
    ML --> XGB[XGBoost]
    ML --> TF[TensorFlow/Keras]
    ML --> PT[PyTorch]
    ML --> HF[Transformers]
    
    Data --> Pandas[Pandas]
    Data --> Numpy[NumPy]
    Data --> CV2[OpenCV]
    
    SKL --> P1[20 Projects]
    XGB --> P2[12 Projects]
    TF --> P3[10 Projects]
    PT --> P4[8 Projects]
    HF --> P5[6 Projects]
    
    style Python fill:#3776ab,stroke:#fff,color:#fff
    style FastAPI fill:#009688,stroke:#fff,color:#fff
    style ML fill:#ff6f00,stroke:#fff,color:#fff
```

---

## Architecture Diagrams

### Standard Microservice Architecture (ASCII)

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                             │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│   │  Web App    │  │  Mobile App │  │  Other APIs │           │
│   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘           │
└──────────┼─────────────────┼────────────────┼──────────────────┘
           │                 │                │
           └─────────────────┼────────────────┘
                             │
                    HTTP/HTTPS REST API
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                      API GATEWAY (Optional)                       │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  - Load Balancing                                          │  │
│  │  - Rate Limiting                                           │  │
│  │  - Authentication                                          │  │
│  │  - Request Routing                                         │  │
│  └────────────────────────────────────────────────────────────┘  │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                      FASTAPI APPLICATION                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                    Endpoint Layer                          │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │  │
│  │  │   GET    │  │  POST    │  │  POST    │  │   GET    │  │  │
│  │  │    /     │  │ /predict │  │ /predict │  │  /model  │  │  │
│  │  │          │  │          │  │  /batch  │  │  /info   │  │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │  │
│  └─────────────────────────┬────────────────────────────────────┘  │
│                            │                                       │
│  ┌─────────────────────────▼────────────────────────────────────┐ │
│  │              Validation Layer (Pydantic)                     │ │
│  │  - Type checking                                             │ │
│  │  - Data validation                                           │ │
│  │  - Schema enforcement                                        │ │
│  └─────────────────────────┬────────────────────────────────────┘ │
│                            │                                       │
│  ┌─────────────────────────▼────────────────────────────────────┐ │
│  │              Business Logic Layer                            │ │
│  │  - Input preprocessing                                       │ │
│  │  - Feature engineering                                       │ │
│  │  - Model inference                                           │ │
│  │  - Post-processing                                           │ │
│  └─────────────────────────┬────────────────────────────────────┘ │
│                            │                                       │
│  ┌─────────────────────────▼────────────────────────────────────┐ │
│  │              ML Model Layer                                  │ │
│  │  ┌────────────────┐  ┌────────────────┐                     │ │
│  │  │  Trained Model │  │  Preprocessors │                     │ │
│  │  │  (In Memory)   │  │  (Scalers, etc)│                     │ │
│  │  └────────────────┘  └────────────────┘                     │ │
│  └──────────────────────────────────────────────────────────────┘ │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            │ Load/Save
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│                    PERSISTENT STORAGE                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Model Files  │  │ Training Data│  │     Logs     │          │
│  │ (.pkl, .h5)  │  │ (CSV, DB)    │  │  (app.log)   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└──────────────────────────────────────────────────────────────────┘
```

### ML Training Pipeline (Mermaid)

```mermaid
flowchart TD
    Start([Start Training]) --> Load[Load Raw Data]
    Load --> EDA[Exploratory Data Analysis]
    
    EDA --> Clean[Data Cleaning]
    Clean --> Split[Train/Test Split]
    
    Split --> FE[Feature Engineering]
    FE --> Scale[Feature Scaling]
    
    Scale --> Balance{Balanced<br/>Classes?}
    Balance -->|No| SMOTE[Apply SMOTE/<br/>Class Weights]
    Balance -->|Yes| Build[Build Model]
    SMOTE --> Build
    
    Build --> Compile[Compile Model]
    Compile --> Train[Train Model]
    
    Train --> Validate[Validation]
    Validate --> Improve{Performance<br/>Good?}
    
    Improve -->|No| Tune[Hyperparameter<br/>Tuning]
    Tune --> Train
    
    Improve -->|Yes| Test[Test on<br/>Hold-out Set]
    
    Test --> Meets{Meets<br/>Threshold?}
    Meets -->|No| Retrain[Collect More Data/<br/>Change Architecture]
    Retrain --> Load
    
    Meets -->|Yes| Save[Save Model]
    Save --> Deploy[Deploy to<br/>Production]
    Deploy --> End([End])
    
    style Start fill:#90EE90
    style End fill:#90EE90
    style Improve fill:#FFD700
    style Meets fill:#FFD700
```

### Prediction Pipeline (Mermaid)

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Validator
    participant Preprocessor
    participant Model
    participant Postprocessor
    
    Client->>API: POST /predict (JSON)
    API->>Validator: Validate Request
    
    alt Invalid Request
        Validator-->>API: Validation Error
        API-->>Client: 422 Unprocessable Entity
    else Valid Request
        Validator-->>API: Valid Data
        API->>Preprocessor: Process Input
        
        Preprocessor->>Preprocessor: Clean Data
        Preprocessor->>Preprocessor: Transform Features
        Preprocessor->>Preprocessor: Scale/Normalize
        
        Preprocessor->>Model: Feature Vector
        Model->>Model: Forward Pass
        Model->>Model: Calculate Probabilities
        
        Model->>Postprocessor: Raw Predictions
        Postprocessor->>Postprocessor: Apply Threshold
        Postprocessor->>Postprocessor: Format Response
        
        Postprocessor->>API: Prediction Result
        API-->>Client: 200 OK (JSON Response)
    end
```

---

## Data Flow Diagrams

### NLP Text Processing Flow (ASCII)

```
┌──────────────────┐
│   Raw Text       │
│  "Free money!!!" │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Lowercase       │
│  "free money!!!" │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Remove Punct    │
│  "free money"    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Tokenization    │
│  ["free","money"]│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Remove Stopwords│
│  ["free","money"]│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Stemming/Lemma  │
│  ["free","money"]│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Vectorization   │
│  TF-IDF/Embeddings│
│  [0.7, 0.9, ...]│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  ML Model        │
│  Classification  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Prediction      │
│  "SPAM" (95%)    │
└──────────────────┘
```

### Computer Vision Processing Flow (ASCII)

```
┌──────────────────┐
│  Input Image     │
│  (Any Size/Format)│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Decode          │
│  Base64 → Binary │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Load Image      │
│  PIL/OpenCV      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Resize          │
│  → (224,224,3)   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Normalize       │
│  [0,255] → [0,1] │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Add Batch Dim   │
│  (H,W,C) → (1,H,W,C)│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  CNN Forward     │
│  Conv→Pool→Dense │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Softmax         │
│  Class Probs     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Top-K Selection │
│  "cat" (92%)     │
│  "dog" (5%)      │
└──────────────────┘
```

### Time Series Forecasting Flow (Mermaid)

```mermaid
graph LR
    A[Historical Data] --> B[Stationarity Check]
    B --> C{Stationary?}
    C -->|No| D[Differencing/<br/>Transformation]
    D --> B
    C -->|Yes| E[Feature Engineering]
    
    E --> F[Lag Features]
    E --> G[Rolling Statistics]
    E --> H[Seasonal Features]
    
    F --> I[Model Training]
    G --> I
    H --> I
    
    I --> J[ARIMA/Prophet/<br/>LSTM]
    J --> K[Validation]
    
    K --> L{Accurate?}
    L -->|No| M[Tune Parameters]
    M --> I
    L -->|Yes| N[Forecast Future]
    
    N --> O[Confidence Intervals]
    O --> P[Predictions]
    
    style A fill:#e1f5ff
    style P fill:#c8e6c9
```

---

## Learning Path Diagrams

### Complete Learning Journey (Mermaid)

```mermaid
graph TB
    Start([Start Here]) --> Basics[Learn Python Basics]
    Basics --> Stats[Statistics & Math Fundamentals]
    
    Stats --> Track{Choose Track}
    
    Track -->|NLP| NLP_Path[NLP Track]
    Track -->|CV| CV_Path[Computer Vision Track]
    Track -->|Analytics| Analytics_Path[Business Analytics Track]
    Track -->|Full-Stack| FS_Path[Full-Stack ML Track]
    
    NLP_Path --> NLP_B[spam-classifier<br/>sentiment-service]
    NLP_B --> NLP_I[fake-news-detector<br/>resume-analyzer]
    NLP_I --> NLP_A[chatbot-api<br/>summarization-api]
    
    CV_Path --> CV_B[digit-recognition]
    CV_B --> CV_I[image-classification<br/>quality-control-cv]
    CV_I --> CV_A[face-recognition]
    
    Analytics_Path --> A_B[churn-prediction<br/>customer-segmentation]
    A_B --> A_I[lead-scoring<br/>sales-forecasting]
    A_I --> A_A[clv-predictor<br/>price-optimization]
    
    FS_Path --> FS_B[4 Beginner Projects]
    FS_B --> FS_I[6 Intermediate Projects]
    FS_I --> FS_A[3 Advanced Projects]
    
    NLP_A --> Expert[Expert Level]
    CV_A --> Expert
    A_A --> Expert
    FS_A --> Expert
    
    Expert --> Portfolio[Build Portfolio]
    Portfolio --> Job[Job Ready!]
    
    style Start fill:#4CAF50,color:#fff
    style Job fill:#FF9800,color:#fff
    style Expert fill:#9C27B0,color:#fff
```

### Project Prerequisites Map (ASCII)

```
LEVEL 1: Prerequisites
├── Python Programming
├── Basic Statistics
├── Linear Algebra Basics
└── Command Line Usage

↓

LEVEL 2: Beginner Projects (Build Foundation)
├── spam-classifier ────────► Learn: Classification, NLP basics
├── house-price-api ────────► Learn: Regression, feature engineering
├── loan-eligibility ───────► Learn: Binary classification
└── sentiment-service ──────► Learn: Pre-trained models

↓

LEVEL 3: Intermediate Projects (Core Skills)
├── churn-prediction ───────► Requires: Classification basics
│   └── Learn: XGBoost, SHAP, imbalanced data
│
├── image-classification ───► Requires: Neural network basics
│   └── Learn: CNNs, transfer learning
│
├── movie-recommender ──────► Requires: Linear algebra
│   └── Learn: Collaborative filtering, matrix factorization
│
└── credit-card-fraud ──────► Requires: Classification
    └── Learn: Anomaly detection, precision-recall tradeoff

↓

LEVEL 4: Advanced Projects (Specialization)
├── chatbot-api ────────────► Requires: NLP intermediate
│   └── Learn: Transformers, BERT, GPT, dialogue
│
├── face-recognition ───────► Requires: CV intermediate
│   └── Learn: Siamese networks, embeddings
│
└── auto-retraining ────────► Requires: Multiple ML projects
    └── Learn: MLOps, monitoring, automation

↓

LEVEL 5: Expert (Production Systems)
└── Build complex, multi-model systems
```

---

## Technology Stack Diagrams

### Complete Technology Ecosystem (ASCII)

```
                    ┌─────────────────────────────────┐
                    │     CLIENT APPLICATIONS         │
                    │  Web • Mobile • Desktop • CLI   │
                    └────────────┬────────────────────┘
                                 │
                    ┌────────────▼────────────────────┐
                    │      API LAYER                  │
                    │  ┌──────────┐  ┌─────────────┐ │
                    │  │ FastAPI  │  │  Uvicorn    │ │
                    │  └──────────┘  └─────────────┘ │
                    │  ┌──────────┐  ┌─────────────┐ │
                    │  │ Pydantic │  │  CORS       │ │
                    │  └──────────┘  └─────────────┘ │
                    └────────────┬────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
┌───────▼─────────┐    ┌────────▼────────┐    ┌─────────▼────────┐
│  CLASSICAL ML   │    │  DEEP LEARNING  │    │  SPECIALIZED     │
│                 │    │                 │    │                  │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌──────────────┐│
│ │Scikit-learn │ │    │ │ TensorFlow  │ │    │ │ SHAP         ││
│ └─────────────┘ │    │ └─────────────┘ │    │ └──────────────┘│
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌──────────────┐│
│ │  XGBoost    │ │    │ │  PyTorch    │ │    │ │ Prophet      ││
│ └─────────────┘ │    │ └─────────────┘ │    │ └──────────────┘│
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌──────────────┐│
│ │  LightGBM   │ │    │ │Transformers │ │    │ │ OpenCV       ││
│ └─────────────┘ │    │ └─────────────┘ │    │ └──────────────┘│
└─────────────────┘    └─────────────────┘    └──────────────────┘
        │                        │                        │
        └────────────────────────┼────────────────────────┘
                                 │
                    ┌────────────▼────────────────────┐
                    │      DATA LAYER                 │
                    │  ┌──────────┐  ┌─────────────┐ │
                    │  │  Pandas  │  │   NumPy     │ │
                    │  └──────────┘  └─────────────┘ │
                    │  ┌──────────┐  ┌─────────────┐ │
                    │  │  Pillow  │  │  Datasets   │ │
                    │  └──────────┘  └─────────────┘ │
                    └────────────┬────────────────────┘
                                 │
                    ┌────────────▼────────────────────┐
                    │    STORAGE & DEPLOYMENT         │
                    │  ┌──────────┐  ┌─────────────┐ │
                    │  │  Docker  │  │ Kubernetes  │ │
                    │  └──────────┘  └─────────────┘ │
                    │  ┌──────────┐  ┌─────────────┐ │
                    │  │   S3     │  │  Postgres   │ │
                    │  └──────────┘  └─────────────┘ │
                    └─────────────────────────────────┘
```

### ML Algorithm Selection Tree (Mermaid)

```mermaid
graph TD
    Start{What's your<br/>problem type?}
    
    Start -->|Predict Number| Reg[Regression]
    Start -->|Predict Category| Class[Classification]
    Start -->|Find Groups| Clust[Clustering]
    Start -->|Recommend Items| Rec[Recommendation]
    
    Reg --> RegQ{Data characteristics?}
    RegQ -->|Linear| LR[Linear Regression<br/>house-price-api]
    RegQ -->|Time Series| TS[ARIMA/Prophet<br/>sales-forecasting]
    RegQ -->|Complex| XGBReg[XGBoost<br/>clv-predictor]
    
    Class --> ClassQ{What type?}
    ClassQ -->|Text| TextC{Simple or Complex?}
    TextC -->|Simple| NB[Naive Bayes<br/>spam-classifier]
    TextC -->|Complex| BERT[Transformers<br/>chatbot-api]
    
    ClassQ -->|Image| ImgC{Dataset size?}
    ImgC -->|Small| CNN[Custom CNN<br/>digit-recognition]
    ImgC -->|Large| ResNet[ResNet/EfficientNet<br/>image-classification]
    
    ClassQ -->|Tabular| TabC{Tree or Linear?}
    TabC -->|Tree| XGBClass[XGBoost<br/>churn-prediction]
    TabC -->|Linear| LogReg[Logistic Regression<br/>loan-eligibility]
    
    Clust --> ClustQ{Know # clusters?}
    ClustQ -->|Yes| KMeans[K-Means<br/>customer-segmentation]
    ClustQ -->|No| DBSCAN[DBSCAN<br/>customer-segmentation]
    
    Rec --> RecQ{What data?}
    RecQ -->|User-Item| CF[Collaborative Filtering<br/>movie-recommender]
    RecQ -->|Item Features| CB[Content-Based<br/>product-recommender]
    RecQ -->|Both| Hybrid[Hybrid System<br/>recommendation-system]
    
    style Start fill:#FFE0B2
    style Reg fill:#C5E1A5
    style Class fill:#90CAF9
    style Clust fill:#CE93D8
    style Rec fill:#FFAB91
```

### Deployment Architecture Options (ASCII)

```
OPTION 1: Single Container (Beginner Projects)
┌────────────────────────────────────────┐
│          Docker Container              │
│  ┌──────────────────────────────────┐  │
│  │   FastAPI App + ML Model         │  │
│  │   Port: 8000                     │  │
│  └──────────────────────────────────┘  │
└────────────────┬───────────────────────┘
                 │
        Exposed Port 8000


OPTION 2: Multi-Container (Intermediate)
┌────────────────────────────────────────┐
│       Docker Compose                   │
│                                        │
│  ┌──────────────┐  ┌───────────────┐  │
│  │  API Server  │  │  Redis Cache  │  │
│  │  Port: 8000  │  │  Port: 6379   │  │
│  └──────┬───────┘  └───────┬───────┘  │
│         └────────┬──────────┘          │
│                  │                     │
│         ┌────────▼────────┐            │
│         │   Shared Vol    │            │
│         │   (Models)      │            │
│         └─────────────────┘            │
└────────────────────────────────────────┘


OPTION 3: Kubernetes (Production)
┌────────────────────────────────────────────────┐
│             Kubernetes Cluster                 │
│                                                │
│  ┌──────────────────────────────────────────┐ │
│  │         Ingress Controller               │ │
│  │  (Load Balancer + SSL Termination)       │ │
│  └──────────────┬───────────────────────────┘ │
│                 │                              │
│  ┌──────────────▼───────────────────────────┐ │
│  │            Service                       │ │
│  │      (Internal Load Balancer)            │ │
│  └──────────────┬───────────────────────────┘ │
│                 │                              │
│     ┌───────────┼───────────┐                 │
│     │           │           │                 │
│  ┌──▼───┐   ┌──▼───┐   ┌──▼───┐              │
│  │ Pod  │   │ Pod  │   │ Pod  │              │
│  │API+ML│   │API+ML│   │API+ML│              │
│  └──────┘   └──────┘   └──────┘              │
│                                                │
│  ┌──────────────────────────────────────────┐ │
│  │        Persistent Volume                 │ │
│  │      (Shared Model Storage)              │ │
│  └──────────────────────────────────────────┘ │
└────────────────────────────────────────────────┘
```

---

## Model Performance Comparison

### Accuracy vs Speed Tradeoff (ASCII Graph)

```
High Accuracy
    │
100%│                                    ● Transformers (chatbot)
    │                              ● ResNet (face-recognition)
 95%│                        ● XGBoost (churn-prediction)
    │                  ● Random Forest (credit-fraud)
 90%│            ● CNN (image-classification)
    │      ● SVM (loan-eligibility)
 85%│ ● Naive Bayes (spam-classifier)
    │
 80%│
    └─────────────────────────────────────────────────────►
      Fast                                            Slow
      <50ms        100ms         200ms         500ms   >1s