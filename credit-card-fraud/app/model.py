import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os

class FraudDetectionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def generate_synthetic_data(self, n_samples=10000):
        """Generate synthetic credit card fraud data for demonstration"""
        np.random.seed(42)
        
        # Generate features
        data = {
            'transaction_amount': np.random.lognormal(4, 1.5, n_samples),
            'merchant_category': np.random.choice(['grocery', 'gas', 'restaurant', 'retail', 'online', 'atm'], n_samples),
            'card_type': np.random.choice(['credit', 'debit'], n_samples, p=[0.7, 0.3]),
            'transaction_type': np.random.choice(['purchase', 'withdrawal', 'transfer'], n_samples, p=[0.8, 0.15, 0.05]),
            'hour_of_day': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'customer_age': np.random.randint(18, 80, n_samples),
            'account_balance': np.random.normal(5000, 2000, n_samples),
            'previous_transaction_amount': np.random.lognormal(4, 1.2, n_samples),
            'transaction_frequency_24h': np.random.poisson(3, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create fraud labels (imbalanced: ~1% fraud)
        fraud_conditions = (
            (df['transaction_amount'] > df['transaction_amount'].quantile(0.95)) |
            (df['hour_of_day'].isin([2, 3, 4])) & (df['transaction_amount'] > 1000) |
            (df['transaction_frequency_24h'] > 10) & (df['transaction_amount'] > 500) |
            (df['transaction_amount'] > df['account_balance'] * 0.8) |
            (np.random.random(n_samples) < 0.005)  # Random fraud
        )
        
        df['is_fraud'] = fraud_conditions.astype(int)
        
        return df
    
    def preprocess_data(self, df, fit_encoders=False):
        """Preprocess the data"""
        df_processed = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['merchant_category', 'card_type', 'transaction_type']
        
        for col in categorical_cols:
            if fit_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
            else:
                df_processed[col] = self.label_encoders[col].transform(df_processed[col])
        
        # Create additional features
        df_processed['amount_to_balance_ratio'] = df_processed['transaction_amount'] / (df_processed['account_balance'] + 1)
        df_processed['amount_change_ratio'] = df_processed['transaction_amount'] / (df_processed['previous_transaction_amount'] + 1)
        
        # Select features for training
        feature_cols = [
            'transaction_amount', 'merchant_category', 'card_type', 'transaction_type',
            'hour_of_day', 'day_of_week', 'customer_age', 'account_balance',
            'previous_transaction_amount', 'transaction_frequency_24h',
            'amount_to_balance_ratio', 'amount_change_ratio'
        ]
        
        self.feature_names = feature_cols
        
        return df_processed[feature_cols], df_processed['is_fraud'] if 'is_fraud' in df_processed.columns else None
    
    def train_model(self, model_type='random_forest'):
        """Train the fraud detection model"""
        # Generate training data
        print("Generating synthetic training data...")
        df = self.generate_synthetic_data(50000)
        
        # Preprocess data
        print("Preprocessing data...")
        X, y = self.preprocess_data(df, fit_encoders=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply SMOTE to handle class imbalance
        print("Applying SMOTE for class imbalance...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        
        # Train model
        print(f"Training {model_type} model...")
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        else:  # logistic_regression
            self.model = LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
        
        self.model.fit(X_train_balanced, y_train_balanced)
        
        # Evaluate model
        print("Evaluating model...")
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        # Store evaluation metrics
        self.metrics = {
            'accuracy': self.model.score(X_test_scaled, y_test),
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'model_type': model_type,
            'training_samples': len(X_train_balanced)
        }
        
        return self.metrics
    
    def predict(self, transaction_data):
        """Make fraud prediction"""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Convert to DataFrame if dict
        if isinstance(transaction_data, dict):
            transaction_data = pd.DataFrame([transaction_data])
        
        # Preprocess
        X_processed, _ = self.preprocess_data(transaction_data, fit_encoders=False)
        X_scaled = self.scaler.transform(X_processed)
        
        # Make prediction
        fraud_probability = self.model.predict_proba(X_scaled)[:, 1][0]
        is_fraud = fraud_probability > 0.5
        
        # Determine confidence level
        if fraud_probability > 0.8:
            confidence = "High"
        elif fraud_probability > 0.6:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Identify risk factors
        risk_factors = []
        if transaction_data['transaction_amount'].iloc[0] > 1000:
            risk_factors.append("High transaction amount")
        if transaction_data['hour_of_day'].iloc[0] in [2, 3, 4]:
            risk_factors.append("Unusual transaction time")
        if transaction_data['transaction_frequency_24h'].iloc[0] > 5:
            risk_factors.append("High transaction frequency")
        if transaction_data['transaction_amount'].iloc[0] > transaction_data['account_balance'].iloc[0] * 0.5:
            risk_factors.append("High amount to balance ratio")
        
        return {
            'fraud_probability': float(fraud_probability),
            'is_fraud': bool(is_fraud),
            'confidence': confidence,
            'risk_factors': risk_factors
        }
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'metrics': self.metrics
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.metrics = model_data['metrics']
        print(f"Model loaded from {filepath}")

# Training script
if __name__ == "__main__":
    # Initialize and train model
    fraud_model = FraudDetectionModel()
    
    # Train with Random Forest
    print("Training Random Forest model...")
    metrics_rf = fraud_model.train_model('random_forest')
    fraud_model.save_model('fraud_detection_model_rf.pkl')
    
    print("\nModel training completed!")
    print(f"Accuracy: {metrics_rf['accuracy']:.4f}")
    print(f"AUC-ROC: {metrics_rf['auc_roc']:.4f}")