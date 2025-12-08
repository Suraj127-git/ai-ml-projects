import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lifetimes import BetaGeoFitter, GammaGammaFitter
    LIFETIMES_AVAILABLE = True
except ImportError:
    LIFETIMES_AVAILABLE = False

class CLVModel:
    """
    Customer Lifetime Value Prediction Model
    Supports both XGBoost regression and BG-NBD (Buy Till You Die) models
    """
    
    def __init__(self):
        self.xgboost_model = None
        self.bg_nbd_model = None
        self.gamma_gamma_model = None
        self.model_performance = {}
        self.feature_names = [
            'recency', 'frequency', 'monetary_value', 'tenure',
            'avg_order_value', 'days_between_purchases', 'total_orders'
        ]
        
    def generate_synthetic_customer_data(self, n_customers: int = 1000) -> pd.DataFrame:
        """Generate synthetic customer transaction data for training/testing"""
        np.random.seed(42)
        
        customers = []
        current_date = datetime.now()
        
        for i in range(n_customers):
            customer_id = f"CUST_{i:04d}"
            
            # Generate purchase behavior
            total_orders = np.random.poisson(5) + 1  # At least 1 order
            if total_orders > 20:
                total_orders = 20  # Cap at 20 orders
                
            # Generate inter-purchase times (exponential distribution)
            days_between_purchases = np.random.exponential(30)  # Average 30 days
            
            # Calculate tenure (time since first purchase)
            max_tenure = min(365 * 3, int(days_between_purchases * total_orders * 1.5))
            tenure = np.random.randint(30, max_tenure + 1)
            
            # Calculate recency (days since last purchase)
            recency = min(tenure, int(np.random.exponential(60)))
            
            # Generate monetary values
            avg_order_value = np.random.lognormal(4, 1)  # Log-normal distribution
            total_spent = avg_order_value * total_orders
            
            # Generate customer demographics
            age = np.random.randint(18, 80)
            gender = np.random.choice(['M', 'F', 'Other'])
            country = np.random.choice(['USA', 'UK', 'Canada', 'Germany', 'France'])
            
            customers.append({
                'customer_id': customer_id,
                'recency': recency,
                'frequency': total_orders,
                'monetary_value': total_spent,
                'tenure': tenure,
                'avg_order_value': avg_order_value,
                'days_between_purchases': days_between_purchases,
                'total_orders': total_orders,
                'age': age,
                'gender': gender,
                'country': country,
                'clv': total_spent * (1 + np.random.normal(0.5, 0.2))  # Future value multiplier
            })
        
        return pd.DataFrame(customers)
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for modeling"""
        feature_df = df.copy()
        
        # Create additional features
        feature_df['purchase_frequency'] = feature_df['frequency'] / (feature_df['tenure'] / 30)  # purchases per month
        feature_df['value_per_day'] = feature_df['monetary_value'] / feature_df['tenure']
        feature_df['recency_score'] = 1 / (feature_df['recency'] + 1)  # Inverse recency
        
        return feature_df
    
    def train_xgboost_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train XGBoost regression model for CLV prediction"""
        if not XGBOOST_AVAILABLE:
            raise RuntimeError("XGBoost not available. Please install it: pip install xgboost")
        
        print("Training XGBoost CLV model...")
        
        # Prepare features
        feature_df = self.prepare_features(df)
        
        # Select features for training
        X = feature_df[self.feature_names + ['purchase_frequency', 'value_per_day', 'recency_score']]
        y = feature_df['clv']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train XGBoost model
        self.xgboost_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            objective='reg:squarederror'
        )
        
        self.xgboost_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.xgboost_model.predict(X_test)
        
        performance = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2_score': r2_score(y_test, y_pred),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        self.model_performance['xgboost'] = performance
        print(f"XGBoost model training completed! RMSE: {performance['rmse']:.2f}, R²: {performance['r2_score']:.3f}")
        
        return performance
    
    def train_bg_nbd_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train BG-NBD model for customer behavior prediction"""
        if not LIFETIMES_AVAILABLE:
            raise RuntimeError("Lifetimes library not available. Please install it: pip install lifetimes")
        
        print("Training BG-NBD model...")
        
        # Prepare data for BG-NBD (RFM format)
        rfm_df = df.copy()
        rfm_df['frequency'] = rfm_df['frequency'] - 1  # BG-NBD expects frequency = number of repeat purchases
        
        # Train BG-NBD model
        self.bg_nbd_model = BetaGeoFitter(penalizer_coef=0.01)
        self.bg_nbd_model.fit(rfm_df['frequency'], rfm_df['recency'], rfm_df['tenure'])
        
        # Train Gamma-Gamma model for monetary value
        self.gamma_gamma_model = GammaGammaFitter(penalizer_coef=0.01)
        
        # Only fit Gamma-Gamma on customers with frequency > 0
        mask = rfm_df['frequency'] > 0
        if mask.sum() > 0:
            self.gamma_gamma_model.fit(rfm_df.loc[mask, 'frequency'], rfm_df.loc[mask, 'monetary_value'])
        
        performance = {
            'training_samples': len(df),
            'model_converged': True,
            'bg_nbd_params': {
                'alpha': self.bg_nbd_model.params_['alpha'],
                'beta': self.bg_nbd_model.params_['beta'],
                'gamma': self.bg_nbd_model.params_['gamma'],
                'delta': self.bg_nbd_model.params_['delta']
            }
        }
        
        if self.gamma_gamma_model and hasattr(self.gamma_gamma_model, 'params_'):
            performance['gamma_gamma_params'] = {
                'p': self.gamma_gamma_model.params_['p'],
                'q': self.gamma_gamma_model.params_['q'],
                'v': self.gamma_gamma_model.params_['v']
            }
        
        self.model_performance['bg_nbd'] = performance
        print("BG-NBD model training completed!")
        
        return performance
    
    def predict_clv_xgboost(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict CLV using XGBoost model"""
        if self.xgboost_model is None:
            raise RuntimeError("XGBoost model not trained. Please train the model first.")
        
        # Prepare input data
        features = [
            customer_data['recency'],
            customer_data['frequency'],
            customer_data['monetary_value'],
            customer_data['tenure'],
            customer_data['avg_order_value'],
            customer_data['days_between_purchases'],
            customer_data['total_orders']
        ]
        
        # Add engineered features
        purchase_frequency = customer_data['frequency'] / (customer_data['tenure'] / 30)
        value_per_day = customer_data['monetary_value'] / customer_data['tenure']
        recency_score = 1 / (customer_data['recency'] + 1)
        
        features.extend([purchase_frequency, value_per_day, recency_score])
        
        # Make prediction
        X = np.array(features).reshape(1, -1)
        predicted_clv = self.xgboost_model.predict(X)[0]
        
        return {
            'predicted_clv': float(predicted_clv),
            'model_used': 'xgboost',
            'confidence_score': 0.85  # Placeholder confidence
        }
    
    def predict_clv_bg_nbd(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict CLV using BG-NBD model"""
        if self.bg_nbd_model is None:
            raise RuntimeError("BG-NBD model not trained. Please train the model first.")
        
        frequency = customer_data['frequency'] - 1  # Convert to repeat purchases
        recency = customer_data['recency']
        tenure = customer_data['tenure']
        monetary_value = customer_data['monetary_value']
        
        # Predict purchases for different time periods
        purchases_30_days = self.bg_nbd_model.conditional_expected_number_of_purchases_up_to_time(
            30, frequency, recency, tenure
        )
        purchases_90_days = self.bg_nbd_model.conditional_expected_number_of_purchases_up_to_time(
            90, frequency, recency, tenure
        )
        purchases_365_days = self.bg_nbd_model.conditional_expected_number_of_purchases_up_to_time(
            365, frequency, recency, tenure
        )
        
        # Estimate monetary value using Gamma-Gamma model
        if self.gamma_gamma_model and frequency > 0:
            predicted_avg_order_value = self.gamma_gamma_model.conditional_expected_average_profit(
                frequency, monetary_value
            )
        else:
            predicted_avg_order_value = monetary_value / customer_data['frequency']
        
        # Calculate CLV (simplified)
        predicted_clv = predicted_avg_order_value * purchases_365_days
        
        return {
            'predicted_clv': float(predicted_clv),
            'model_used': 'bg_nbd',
            'expected_purchases_next_30_days': float(purchases_30_days),
            'expected_purchases_next_90_days': float(purchases_90_days),
            'expected_purchases_next_365_days': float(purchases_365_days),
            'confidence_score': 0.80  # Placeholder confidence
        }
    
    def save_model(self, filepath: str, model_type: str):
        """Save trained model"""
        model_data = {
            'model_type': model_type,
            'xgboost_model': self.xgboost_model,
            'bg_nbd_model': self.bg_nbd_model,
            'gamma_gamma_model': self.gamma_gamma_model,
            'model_performance': self.model_performance,
            'feature_names': self.feature_names,
            'saved_at': datetime.now().isoformat()
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.xgboost_model = model_data['xgboost_model']
        self.bg_nbd_model = model_data['bg_nbd_model']
        self.gamma_gamma_model = model_data['gamma_gamma_model']
        self.model_performance = model_data['model_performance']
        self.feature_names = model_data['feature_names']
        print(f"Model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about trained models"""
        info = {
            'xgboost_available': XGBOOST_AVAILABLE,
            'lifetimes_available': LIFETIMES_AVAILABLE,
            'models_trained': {
                'xgboost': self.xgboost_model is not None,
                'bg_nbd': self.bg_nbd_model is not None,
                'gamma_gamma': self.gamma_gamma_model is not None
            },
            'model_performance': self.model_performance
        }
        return info