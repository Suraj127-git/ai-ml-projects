"""
Fake News Detector Training Script
This script demonstrates how to train a fake news detection model
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def preprocess_text(text):
    """Preprocess text for fake news detection"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Join tokens back
    return ' '.join(tokens)

def create_sample_dataset():
    """Create a sample dataset for demonstration"""
    logger.info("Creating sample dataset...")
    
    # Sample fake news (label = 1)
    fake_news = [
        "BREAKING: Scientists discover that eating chocolate cures cancer completely!",
        "SHOCKING: Government hiding truth about alien invasion planned for next month",
        "Amazing breakthrough: New pill makes you lose 50 pounds in one week without exercise",
        "URGENT: World leaders meeting in secret to implement global mind control",
        "Scientists reveal shocking truth: Earth is actually flat and space is fake",
        "Health alert: Common household item causes cancer - throw it away immediately",
        "Financial miracle: Invest $100 and become millionaire in 30 days guaranteed",
        "Weather control: Government using secret technology to create hurricanes",
        "Time travel proven: Man returns from year 2050 with shocking predictions",
        "Medical conspiracy: Doctors hiding simple cure for all diseases"
    ]
    
    # Sample real news (label = 0)
    real_news = [
        "Local city council approves new budget for road improvements",
        "Weather forecast: Partly cloudy with chance of rain this weekend",
        "School district announces new after-school programs for students",
        "Local restaurant celebrates 10th anniversary with community event",
        "Traffic alert: Main street closed for construction next week",
        "Library hosts reading program for children during summer break",
        "Community center offers free fitness classes for seniors",
        "Local high school wins regional science competition",
        "City parks department plants new trees in downtown area",
        "Chamber of commerce hosts monthly business networking event"
    ]
    
    # Create DataFrame
    texts = fake_news + real_news
    labels = [1] * len(fake_news) + [0] * len(real_news)
    
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"Created dataset with {len(df)} samples")
    logger.info(f"Fake news samples: {sum(df['label'] == 1)}")
    logger.info(f"Real news samples: {sum(df['label'] == 0)}")
    
    return df

def train_tfidf_model(X_train, y_train):
    """Train TF-IDF + Logistic Regression model"""
    logger.info("Training TF-IDF model...")
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2
    )
    
    # Fit and transform training data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Train logistic regression
    classifier = LogisticRegression(random_state=42, max_iter=1000)
    classifier.fit(X_train_tfidf, y_train)
    
    logger.info("TF-IDF model training completed")
    
    return vectorizer, classifier

def evaluate_model(classifier, vectorizer, X_test, y_test):
    """Evaluate the trained model"""
    logger.info("Evaluating model...")
    
    # Transform test data
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Make predictions
    y_pred = classifier.predict(X_test_tfidf)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Real', 'Fake'])
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    logger.info(f"Model Accuracy: {accuracy:.4f}")
    logger.info("Classification Report:")
    logger.info(report)
    logger.info("Confusion Matrix:")
    logger.info(conf_matrix)
    
    return accuracy, report, conf_matrix

def save_models(vectorizer, classifier, accuracy):
    """Save trained models"""
    logger.info("Saving models...")
    
    # Save vectorizer
    joblib.dump(vectorizer, '../app/models/tfidf_vectorizer.pkl')
    logger.info("Vectorizer saved to ../app/models/tfidf_vectorizer.pkl")
    
    # Save classifier
    joblib.dump(classifier, '../app/models/tfidf_classifier.pkl')
    logger.info("Classifier saved to ../app/models/tfidf_classifier.pkl")
    
    # Save model info
    model_info = {
        'accuracy': accuracy,
        'model_type': 'tfidf_logistic',
        'features': ['TF-IDF vectors', 'n-grams', 'stop words removal'],
        'description': 'TF-IDF vectorization with Logistic Regression for fake news detection'
    }
    
    joblib.dump(model_info, '../app/models/model_info.pkl')
    logger.info("Model info saved to ../app/models/model_info.pkl")

def main():
    """Main training function"""
    logger.info("Starting Fake News Detector training...")
    
    # Create sample dataset
    df = create_sample_dataset()
    
    # Preprocess texts
    logger.info("Preprocessing texts...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], 
        df['label'], 
        test_size=0.2, 
        random_state=42,
        stratify=df['label']
    )
    
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")
    
    # Train model
    vectorizer, classifier = train_tfidf_model(X_train, y_train)
    
    # Evaluate model
    accuracy, report, conf_matrix = evaluate_model(classifier, vectorizer, X_test, y_test)
    
    # Create models directory
    import os
    os.makedirs('../app/models', exist_ok=True)
    
    # Save models
    save_models(vectorizer, classifier, accuracy)
    
    logger.info("Training completed successfully!")
    logger.info(f"Final model accuracy: {accuracy:.4f}")
    
    # Test the model with some examples
    logger.info("\nTesting the model with examples:")
    test_examples = [
        "Breaking news: Scientists discover cure for all diseases with simple household ingredient!",
        "Local school board approves new curriculum for mathematics education",
        "Shocking revelation: Government hiding evidence of alien contact",
        "Weather forecast: Sunny and warm conditions expected this weekend"
    ]
    
    for example in test_examples:
        processed_example = preprocess_text(example)
        example_tfidf = vectorizer.transform([processed_example])
        prediction = classifier.predict(example_tfidf)[0]
        probability = classifier.predict_proba(example_tfidf)[0]
        
        label = "FAKE" if prediction == 1 else "REAL"
        confidence = max(probability)
        
        logger.info(f"Text: {example}")
        logger.info(f"Prediction: {label} (confidence: {confidence:.4f})")
        logger.info("---")

if __name__ == "__main__":
    main()