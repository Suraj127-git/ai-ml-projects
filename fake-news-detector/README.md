# Fake News Detector API

A FastAPI-based microservice for detecting fake news using various NLP techniques including TF-IDF with Logistic Regression and transformer-based models like BERT and RoBERTa.

## Features

- **Multiple Models**: Support for TF-IDF + Logistic Regression, BERT, and RoBERTa models
- **Real-time Prediction**: Single text prediction endpoint
- **Batch Processing**: Process multiple news articles in one request
- **Model Management**: Load and manage different models dynamically
- **Health Monitoring**: Health check endpoint for service monitoring
- **Comprehensive Scoring**: Returns prediction confidence and class probabilities

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download NLTK data (if not already done):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Usage

### Running the API

```bash
uvicorn app.main:app --reload --port 8009
```

### API Endpoints

#### 1. Single Text Prediction
**POST** `/predict`

Request:
```json
{
  "text": "Breaking news: Scientists discover that chocolate cures cancer completely!",
  "title": "Chocolate Cancer Cure",
  "model_type": "tfidf_logistic"
}
```

Response:
```json
{
  "text": "Breaking news: Scientists discover that chocolate cures cancer completely!",
  "title": "Chocolate Cancer Cure",
  "prediction": "FAKE",
  "confidence": 0.89,
  "model_type": "tfidf_logistic",
  "probabilities": {
    "REAL": 0.11,
    "FAKE": 0.89
  }
}
```

#### 2. Batch Prediction
**POST** `/predict/batch`

Request:
```json
{
  "texts": [
    {
      "text": "Local school board approves new curriculum changes",
      "title": "School Board Update"
    },
    {
      "text": "Shocking revelation: Government hiding alien contact evidence",
      "title": "Alien Conspiracy"
    }
  ],
  "model_type": "tfidf_logistic"
}
```

#### 3. Available Models
**GET** `/models`

Returns information about all available models including descriptions and accuracy metrics.

#### 4. Health Check
**GET** `/health`

Returns service health status and loaded models.

#### 5. Load Model
**GET** `/models/{model_type}/load`

Loads a specific model into memory.

## Model Types

### 1. TF-IDF + Logistic Regression (`tfidf_logistic`)
- **Description**: Traditional machine learning approach using TF-IDF vectorization
- **Features**: N-grams, stop words removal, logistic regression classifier
- **Accuracy**: ~85% (on sample data)
- **Speed**: Fast inference
- **Memory**: Low memory footprint

### 2. BERT (`bert`)
- **Description**: BERT base model fine-tuned for text classification
- **Features**: Transformer architecture, contextual embeddings
- **Accuracy**: ~92% (estimated)
- **Speed**: Moderate inference speed
- **Memory**: Higher memory requirements

### 3. RoBERTa (`roberta`)
- **Description**: Robustly optimized BERT approach
- **Features**: Improved training procedure over BERT
- **Accuracy**: ~94% (estimated)
- **Speed**: Moderate inference speed
- **Memory**: Higher memory requirements

## Training

To train your own model:

1. Prepare your dataset with text and labels (1 for fake, 0 for real)
2. Run the training script:
```bash
python notebooks/train_fake_news_detector.py
```

The script will:
- Create a sample dataset (or load your own)
- Preprocess the text data
- Train a TF-IDF + Logistic Regression model
- Evaluate the model performance
- Save the trained model

## Text Preprocessing

The API performs the following preprocessing steps:
- Converts text to lowercase
- Removes special characters and digits
- Tokenizes the text
- Removes stopwords
- Joins tokens back into processed text

## Performance Considerations

- **TF-IDF Model**: Best for production environments requiring fast inference
- **Transformer Models**: Better accuracy but require more computational resources
- **Batch Processing**: Use batch endpoints for processing multiple texts efficiently
- **Model Loading**: Models are loaded on-demand to save memory

## Example Use Cases

1. **Social Media Monitoring**: Automatically flag potentially fake news in social media posts
2. **Content Moderation**: Help content moderators identify suspicious news articles
3. **News Aggregation**: Filter out likely fake news from news feeds
4. **Educational Tools**: Help students identify misinformation
5. **Research**: Analyze patterns in fake vs. real news content

## Limitations

- Models are trained on sample data and should be retrained with real datasets for production use
- Performance may vary on different types of text and domains
- Transformer models require significant computational resources
- The API should be used as a tool to assist human judgment, not replace it

## Future Improvements

- Fine-tune transformer models on specific fake news datasets
- Add more sophisticated feature engineering
- Implement ensemble methods combining multiple models
- Add explainability features to understand model predictions
- Support for multilingual fake news detection
- Integration with fact-checking APIs