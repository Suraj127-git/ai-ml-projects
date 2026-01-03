# 🚀 AI/ML Projects - Quick Reference Guide

## Project Index (36 Projects)

### 🟢 Beginner Projects (7 projects, 8-15 hours each)

| # | Project | Algorithm | Domain | Key Features | Port |
|---|---------|-----------|--------|--------------|------|
| 1 | **spam-classifier** | Naive Bayes | NLP | Email spam detection, TF-IDF | 8000 |
| 2 | **house-price-api** | Linear Regression | Regression | House price prediction | 8001 |
| 3 | **digit-recognition** | CNN | Computer Vision | MNIST digit classification | 8002 |
| 4 | **loan-eligibility** | Logistic Regression | Classification | Loan approval prediction | 8003 |
| 5 | **stock-price-classifier** | Random Forest | Finance | Stock movement prediction | 8004 |
| 6 | **sentiment-service** | VADER/TextBlob | NLP | Sentiment analysis | 8005 |
| 7 | **news-aggregator** | Clustering | NLP | News topic aggregation | 8006 |

### 🟡 Intermediate Projects (12 projects, 15-30 hours each)

| # | Project | Algorithm | Domain | Key Features | Port |
|---|---------|-----------|--------|--------------|------|
| 8 | **churn-prediction** | XGBoost + SHAP | Customer Analytics | Customer retention, explainability | 8007 |
| 9 | **image-classification** | CNN/ResNet | Computer Vision | Multi-architecture, transfer learning | 8008 |
| 10 | **movie-recommender** | Collaborative Filtering | Recommendation | Movie recommendations | 8009 |
| 11 | **fake-news-detector** | Ensemble (RF+XGB) | NLP | Fake news detection | 8010 |
| 12 | **credit-card-fraud** | Isolation Forest/XGB | Fraud Detection | Anomaly detection, imbalanced data | 8011 |
| 13 | **lead-scoring** | Gradient Boosting | Sales | Sales lead prioritization | 8012 |
| 14 | **customer-segmentation** | K-Means/DBSCAN | Marketing | Customer clustering | 8013 |
| 15 | **sales-forecasting** | ARIMA/Prophet | Time Series | Sales prediction | 8014 |
| 16 | **product-recommender** | Hybrid (Content+CF) | E-commerce | Product recommendations | 8015 |
| 17 | **resume-analyzer** | NLP + ML | HR Tech | Resume parsing, skill matching | 8016 |
| 18 | **quality-control-cv** | CNN + Detection | Manufacturing | Defect detection | 8017 |
| 19 | **predictive-maintenance** | Random Forest/LSTM | IoT | Equipment failure prediction | 8018 |

### 🔴 Advanced Projects (17 projects, 30-50 hours each)

| # | Project | Algorithm | Domain | Key Features | Port |
|---|---------|-----------|--------|--------------|------|
| 20 | **face-recognition** | FaceNet/Siamese | Computer Vision | Face embeddings, verification | 8019 |
| 21 | **chatbot-api** | Transformers (BERT/GPT) | Conversational AI | Dialogue management, NLU | 8020 |
| 22 | **summarization-api** | T5/BART | NLP | Text summarization | 8021 |
| 23 | **text-to-sql** | Seq2seq + Schema | NLP to DB | Natural language to SQL | 8022 |
| 24 | **speech-to-text** | Wav2Vec/Whisper | Speech | Audio transcription | 8023 |
| 25 | **demand-forecasting-neural** | LSTM/Transformer | Supply Chain | Deep time series forecasting | 8024 |
| 26 | **energy-consumption-forecasting** | Neural Prophet | Utilities | Energy usage prediction | 8025 |
| 27 | **clv-predictor** | Survival Analysis | Customer Analytics | Customer lifetime value | 8026 |
| 28 | **price-optimization-engine** | Reinforcement Learning | Pricing | Dynamic pricing | 8027 |
| 29 | **supply-chain-optimization** | Linear Programming | Operations | Supply chain optimization | 8028 |
| 30 | **inventory-optimization** | Time Series + Optimization | Logistics | Inventory management | 8029 |
| 31 | **market-basket-analysis** | Apriori/FP-Growth | Retail | Association rule mining | 8030 |
| 32 | **recommendation-system-collaborative** | Neural CF (NCF) | Recommendation | Deep collaborative filtering | 8031 |
| 33 | **image-classification-products** | EfficientNet/ViT | E-commerce CV | Advanced architectures | 8032 |
| 34 | **auto-retraining** | Model Monitoring | MLOps | Automated model retraining | 8033 |
| 35 | **product-demand-forecasting** | Prophet/LSTM | Supply Chain | Product demand prediction | 8034 |
| 36 | **Math** | Various | Education | Mathematical computations | 8035 |

---

## Quick Start Commands

### Universal Setup
```bash
# Clone repository
git clone <repo-url>
cd ai-ml-projects

# Choose a project
cd <project-name>

# Install dependencies
pip install -r requirements.txt

# Start API server
uvicorn app.main:app --reload --port 8000
```

### Docker Deployment
```bash
# Build and run
docker build -t <project-name> .
docker run -d -p 8000:8000 <project-name>

# Check logs
docker logs <container-id>
```

### Testing API
```bash
# Health check
curl http://localhost:8000/

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"key": "value"}'

# Swagger UI
open http://localhost:8000/docs
```

---

## Technology Stack Summary

### Web Framework
- **FastAPI**: All 36 projects
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation

### Machine Learning Libraries

#### Classical ML (20 projects)
- **Scikit-learn**: Classification, regression, clustering
- **XGBoost**: Gradient boosting (12 projects)
- **LightGBM**: Fast gradient boosting (4 projects)

#### Deep Learning (16 projects)
- **TensorFlow/Keras**: CNNs, RNNs (10 projects)
- **PyTorch**: Advanced architectures (8 projects)
- **Transformers (Hugging Face)**: NLP models (6 projects)

#### Specialized
- **SHAP**: Model explainability (8 projects)
- **Prophet**: Time series forecasting (5 projects)
- **OpenCV**: Computer vision (6 projects)
- **NLTK/spaCy**: NLP preprocessing (10 projects)

---

## API Endpoint Patterns

### Standard Endpoints (All Projects)
```
GET  /                    # Health check
GET  /health             # Detailed health status
POST /predict            # Single prediction
POST /predict/batch      # Batch predictions (optional)
GET  /model/info         # Model metadata
```

### Training Endpoints (Optional)
```
POST /train              # Train/retrain model
GET  /model/metrics      # Model performance metrics
GET  /model/feature_importance  # Feature analysis
```

### Request/Response Format
```json
// Request
{
  "data": {...},          // Input features
  "options": {...}        // Optional parameters
}

// Response
{
  "prediction": ...,      // Main prediction
  "confidence": 0.95,     // Confidence score
  "metadata": {...}       // Additional info
}
```

---

## Learning Tracks

### Track 1: NLP Engineer (6 weeks)
```
Week 1: spam-classifier → sentiment-service
Week 2: fake-news-detector
Week 3: resume-analyzer
Week 4-5: chatbot-api
Week 6: summarization-api OR text-to-sql
```

### Track 2: Computer Vision (6 weeks)
```
Week 1: digit-recognition
Week 2-3: image-classification
Week 4: quality-control-cv
Week 5-6: face-recognition
```

### Track 3: Business Analytics (8 weeks)
```
Week 1-2: churn-prediction + customer-segmentation
Week 3: lead-scoring
Week 4-5: clv-predictor
Week 6: sales-forecasting
Week 7-8: price-optimization-engine
```

### Track 4: Full-Stack ML (12 weeks)
```
Weeks 1-2: 4 beginner projects
Weeks 3-8: 6 intermediate projects
Weeks 9-12: 2-3 advanced projects
```

---

## Performance Benchmarks

### Response Times (Average)
```
Fast (<50ms):
- spam-classifier, house-price-api, loan-eligibility

Medium (50-200ms):
- churn-prediction, image-classification, credit-card-fraud

Slow (>200ms):
- chatbot-api, face-recognition, summarization-api
```

### Model Sizes
```
Small (<20MB):
- spam-classifier (12MB)
- house-price-api (5MB)
- loan-eligibility (8MB)

Medium (20-200MB):
- churn-prediction (50MB)
- image-classification (200MB)
- movie-recommender (100MB)

Large (>200MB):
- chatbot-api (2GB)
- face-recognition (500MB)
- summarization-api (2GB)
```

### Accuracy Targets
```
Classification:
- Binary: >90% accuracy
- Multi-class: >85% accuracy
- Imbalanced: >0.90 AUC-ROC

Regression:
- R² score: >0.80
- RMSE: Dataset-dependent

Recommendation:
- Precision@10: >0.30
- NDCG: >0.70
```

---

## Common Issues & Solutions

### Issue: Model Not Found
```bash
# Solution: Train model first
cd notebooks/
jupyter notebook train_model.ipynb
# Run all cells
```

### Issue: Memory Error
```bash
# Solution: Reduce batch size or use cloud
# Edit app/model.py
BATCH_SIZE = 32  # Reduce from 64
```

### Issue: Slow Training
```bash
# Solution: Use GPU or reduce epochs
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Issue: Port Already in Use
```bash
# Solution: Change port or kill process
uvicorn app.main:app --port 8001

# Or kill existing process
lsof -ti:8000 | xargs kill -9  # Unix
netstat -ano | findstr :8000   # Windows
```

### Issue: Dependency Conflicts
```bash
# Solution: Use virtual environment
python -m venv venv
source venv/bin/activate  # Unix
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

---

## Project Difficulty Matrix

### By Technical Complexity
```
Easy (1-2 stars):
★★☆☆☆ spam-classifier, house-price-api, loan-eligibility

Medium (3 stars):
★★★☆☆ churn-prediction, image-classification, movie-recommender

Hard (4-5 stars):
★★★★★ chatbot-api, face-recognition, text-to-sql, auto-retraining
```

### By Domain Knowledge Required
```
Low:
- Most beginner projects
- Standard ML algorithms

Medium:
- Time series projects (forecasting)
- Computer vision projects
- Recommendation systems

High:
- NLP with transformers
- Reinforcement learning
- Optimization problems
- MLOps systems
```

---

## Dataset Information

### Public Datasets Used
```
MNIST: 70,000 handwritten digits (digit-recognition)
CIFAR-10: 60,000 images, 10 classes (image-classification)
MovieLens: 100,000+ movie ratings (movie-recommender)
SMS Spam: 5,572 SMS messages (spam-classifier)
Telco Churn: 7,043 customer records (churn-prediction)
Credit Card Fraud: 284,807 transactions (credit-card-fraud)
```

### Synthetic Data Generation
```
Many projects include synthetic data generators:
- customer-segmentation
- sales-forecasting
- lead-scoring
- inventory-optimization
```

---

## Hardware Requirements

### Minimum (Beginner Projects)
```
CPU: 2 cores, 2.0 GHz
RAM: 4 GB
Storage: 10 GB
GPU: Not required
```

### Recommended (Intermediate Projects)
```
CPU: 4 cores, 2.5 GHz
RAM: 8 GB
Storage: 20 GB
GPU: Optional (speeds up training)
```

### Optimal (Advanced Projects)
```
CPU: 8+ cores, 3.0 GHz
RAM: 16-32 GB
Storage: 50+ GB SSD
GPU: NVIDIA (6GB+ VRAM)
```

### Cloud Alternatives
```
Google Colab: Free GPU/TPU
Kaggle Kernels: Free GPU (30h/week)
AWS SageMaker: Pay-as-you-go
GCP AI Platform: Flexible options
Azure ML: Enterprise features
```

---

## Model Deployment Checklist

### Pre-Deployment
- [ ] Model trained and validated
- [ ] Accuracy meets requirements
- [ ] Dependencies documented
- [ ] Environment variables configured
- [ ] Error handling implemented
- [ ] Logging configured
- [ ] API documentation complete

### Deployment
- [ ] Docker image built and tested
- [ ] Health check endpoint working
- [ ] Load testing completed
- [ ] Monitoring set up
- [ ] Backup/rollback plan ready
- [ ] SSL/HTTPS configured
- [ ] Rate limiting enabled

### Post-Deployment
- [ ] Monitor performance metrics
- [ ] Track prediction accuracy
- [ ] Collect user feedback
- [ ] Plan model updates
- [ ] Document issues and resolutions

---

## Key Metrics to Track

### Technical Metrics
```
Latency: p50, p95, p99 response times
Throughput: Requests per second
Error Rate: 4xx and 5xx errors
Model Accuracy: Test set performance
Resource Usage: CPU, memory, disk
```

### Business Metrics
```
User Adoption: Active users, requests/day
ROI: Cost savings, revenue impact
Accuracy in Production: Real-world performance
Customer Satisfaction: Feedback scores
Time Savings: Manual work reduced
```

---

## Best Practices

### Code Quality
```python
# Use type hints
def predict(features: List[float]) -> Dict[str, Any]:
    pass

# Validate inputs with Pydantic
class PredictionRequest(BaseModel):
    features: List[float]

# Handle errors gracefully
try:
    prediction = model.predict(data)
except Exception as e:
    logger.error(f"Prediction failed: {e}")
    raise HTTPException(status_code=500)
```

### Model Management
```python
# Version your models
MODEL_VERSION = "1.2.0"
MODEL_PATH = f"models/model_v{MODEL_VERSION}.pkl"

# Log predictions for monitoring
logger.info(f"Prediction: {result}, Confidence: {confidence}")

# Implement model fallback
if primary_model_fails:
    use_backup_model()
```

### Security
```python
# Validate input sizes
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_TEXT_LENGTH = 10000

# Sanitize inputs
import bleach
clean_text = bleach.clean(user_input)

# Rate limit requests
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)
```

---

## Useful Resources

### Documentation
- FastAPI: https://fastapi.tiangolo.com/
- Scikit-learn: https://scikit-learn.org/
- TensorFlow: https://www.tensorflow.org/
- PyTorch: https://pytorch.org/
- Hugging Face: https://huggingface.co/docs

### Learning Platforms
- Coursera: ML Specialization
- Fast.ai: Practical Deep Learning
- Kaggle Learn: Free courses
- DataCamp: Interactive tutorials
- YouTube: StatQuest, 3Blue1Brown

### Communities
- r/MachineLearning
- r/learnmachinelearning
- Stack Overflow ML tags
- GitHub ML repositories
- Twitter ML community

---

## Project Comparison Table

| Feature | Beginner | Intermediate | Advanced |
|---------|----------|--------------|----------|
| Setup Time | 10-20 min | 20-40 min | 40-120 min |
| Learning Time | 8-15 hours | 15-30 hours | 30-50 hours |
| Code Complexity | Low | Medium | High |
| ML Concepts | Basic | Moderate | Advanced |
| Model Size | <20MB | 20-200MB | >200MB |
| Response Time | <50ms | 50-200ms | >200ms |
| Resource Usage | Low | Medium | High |
| Production Ready | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

---

## Next Steps

### For Beginners
1. Start with **spam-classifier** (easiest)
2. Move to **house-price-api** (regression)
3. Try **digit-recognition** (computer vision)
4. Complete **sentiment-service** (pre-trained models)

### For Intermediate
1. Master **churn-prediction** (XGBoost + SHAP)
2. Build **image-classification** (CNNs)
3. Create **movie-recommender** (collaborative filtering)
4. Deploy **credit-card-fraud** (imbalanced data)

### For Advanced
1. Implement **chatbot-api** (transformers)
2. Build **face-recognition** (embeddings)
3. Create **text-to-sql** (seq2seq)
4. Deploy **auto-retraining** (MLOps)

---

## Contact & Support

- **Documentation**: See individual project README files
- **Issues**: Report on GitHub
- **Questions**: Use GitHub Discussions
- **Contributions**: Pull requests welcome

---

**Last Updated**: 2024
**Total Projects**: 36
**Total Learning Time**: 300-400 hours
**License**: MIT

🚀 Happy Learning!