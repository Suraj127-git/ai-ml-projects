"""
Training script for CLV Predictor models
This script can be used to train models with custom datasets
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.model import CLVModel

def load_sample_data():
    """Load or generate sample data for training"""
    model = CLVModel()
    
    # Generate synthetic data
    print("Generating synthetic customer data...")
    df = model.generate_synthetic_customer_data(n_customers=2000)
    
    print(f"Generated {len(df)} customer records")
    print("\nData summary:")
    print(df.describe())
    
    return df

def train_clv_models(df):
    """Train both XGBoost and BG-NBD models"""
    model = CLVModel()
    
    # Train XGBoost model
    print("\n" + "="*50)
    print("Training XGBoost Model")
    print("="*50)
    
    try:
        xgboost_performance = model.train_xgboost_model(df)
        print(f"XGBoost training completed!")
        print(f"RMSE: {xgboost_performance['rmse']:.2f}")
        print(f"R² Score: {xgboost_performance['r2_score']:.3f}")
        print(f"MAE: {xgboost_performance['mae']:.2f}")
    except Exception as e:
        print(f"XGBoost training failed: {e}")
        xgboost_performance = None
    
    # Train BG-NBD model
    print("\n" + "="*50)
    print("Training BG-NBD Model")
    print("="*50)
    
    try:
        bg_nbd_performance = model.train_bg_nbd_model(df)
        print(f"BG-NBD training completed!")
        print(f"Training samples: {bg_nbd_performance['training_samples']}")
        print(f"Model converged: {bg_nbd_performance['model_converged']}")
    except Exception as e:
        print(f"BG-NBD training failed: {e}")
        bg_nbd_performance = None
    
    return model, xgboost_performance, bg_nbd_performance

def evaluate_models(model, test_data):
    """Evaluate model performance on test data"""
    print("\n" + "="*50)
    print("Model Evaluation")
    print("="*50)
    
    # Sample a few customers for testing
    sample_customers = test_data.sample(n=5, random_state=42)
    
    for idx, customer in sample_customers.iterrows():
        customer_data = customer.to_dict()
        customer_id = customer_data['customer_id']
        actual_clv = customer_data['clv']
        
        print(f"\nCustomer: {customer_id}")
        print(f"Actual CLV: ${actual_clv:.2f}")
        
        # XGBoost prediction
        if model.xgboost_model is not None:
            try:
                xgboost_pred = model.predict_clv_xgboost(customer_data)
                predicted_clv = xgboost_pred['predicted_clv']
                error = abs(predicted_clv - actual_clv)
                print(f"XGBoost Prediction: ${predicted_clv:.2f} (Error: ${error:.2f})")
            except Exception as e:
                print(f"XGBoost prediction failed: {e}")
        
        # BG-NBD prediction
        if model.bg_nbd_model is not None:
            try:
                bg_nbd_pred = model.predict_clv_bg_nbd(customer_data)
                predicted_clv = bg_nbd_pred['predicted_clv']
                error = abs(predicted_clv - actual_clv)
                print(f"BG-NBD Prediction: ${predicted_clv:.2f} (Error: ${error:.2f})")
                print(f"Expected purchases (30d): {bg_nbd_pred.get('expected_purchases_next_30_days', 0):.2f}")
                print(f"Expected purchases (365d): {bg_nbd_pred.get('expected_purchases_next_365_days', 0):.2f}")
            except Exception as e:
                print(f"BG-NBD prediction failed: {e}")

def save_models(model):
    """Save trained models"""
    print("\n" + "="*50)
    print("Saving Models")
    print("="*50)
    
    # Create models directory
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Save XGBoost model
    if model.xgboost_model is not None:
        xgboost_path = os.path.join(models_dir, "clv_xgboost_model.pkl")
        model.save_model(xgboost_path, "xgboost")
        print(f"XGBoost model saved to {xgboost_path}")
    
    # Save BG-NBD model
    if model.bg_nbd_model is not None:
        bg_nbd_path = os.path.join(models_dir, "clv_bg_nbd_model.pkl")
        model.save_model(bg_nbd_path, "bg_nbd")
        print(f"BG-NBD model saved to {bg_nbd_path}")

def main():
    """Main training function"""
    print("Customer Lifetime Value (CLV) Predictor - Model Training")
    print("="*60)
    print(f"Training started at: {datetime.now()}")
    
    # Load data
    df = load_sample_data()
    
    # Split data for training and testing
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    
    print(f"\nTraining set: {len(train_df)} customers")
    print(f"Test set: {len(test_df)} customers")
    
    # Train models
    model, xgboost_perf, bg_nbd_perf = train_clv_models(train_df)
    
    # Evaluate models
    evaluate_models(model, test_df)
    
    # Save models
    save_models(model)
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Completed at: {datetime.now()}")
    
    # Print final summary
    print("\nTraining Summary:")
    if xgboost_perf:
        print(f"XGBoost - RMSE: {xgboost_perf['rmse']:.2f}, R²: {xgboost_perf['r2_score']:.3f}")
    if bg_nbd_perf:
        print(f"BG-NBD - Training samples: {bg_nbd_perf['training_samples']}, Converged: {bg_nbd_perf['model_converged']}")

if __name__ == "__main__":
    main()