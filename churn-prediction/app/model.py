import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import xgboost as xgb
import shap
import joblib
import warnings
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

class ChurnPredictionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.model_info = {}
        self.shap_explainer = None
        
    def generate_synthetic_data(self, n_samples: int = 5000) -> pd.DataFrame:
        """Generate synthetic customer churn data"""
        np.random.seed(42)
        
        data = []
        for i in range(n_samples):
            customer_id = f"CUST_{i+1:06d}"
            
            # Demographics
            gender = np.random.choice(['Male', 'Female'])
            senior_citizen = np.random.choice([0, 1], p=[0.85, 0.15])
            partner = np.random.choice(['Yes', 'No'], p=[0.6, 0.4])
            dependents = np.random.choice(['Yes', 'No'], p=[0.3, 0.7])
            
            # Service information
            tenure = np.random.randint(1, 73)  # 1 to 72 months
            phone_service = np.random.choice(['Yes', 'No'], p=[0.9, 0.1])
            
            if phone_service == 'Yes':
                multiple_lines = np.random.choice(['Yes', 'No'], p=[0.4, 0.6])
            else:
                multiple_lines = 'No phone service'
            
            internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], p=[0.4, 0.3, 0.3])
            
            # Internet-dependent services
            if internet_service == 'No':
                online_security = online_backup = device_protection = tech_support = streaming_tv = streaming_movies = 'No internet service'
            else:
                online_security = np.random.choice(['Yes', 'No'], p=[0.3, 0.7])
                online_backup = np.random.choice(['Yes', 'No'], p=[0.4, 0.6])
                device_protection = np.random.choice(['Yes', 'No'], p=[0.3, 0.7])
                tech_support = np.random.choice(['Yes', 'No'], p=[0.3, 0.7])
                streaming_tv = np.random.choice(['Yes', 'No'], p=[0.4, 0.6])
                streaming_movies = np.random.choice(['Yes', 'No'], p=[0.4, 0.6])
            
            # Account information
            contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], p=[0.5, 0.3, 0.2])
            paperless_billing = np.random.choice(['Yes', 'No'], p=[0.6, 0.4])
            payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], p=[0.3, 0.2, 0.25, 0.25])
            
            # Financial information
            monthly_charges = np.random.uniform(20, 120)
            
            # Adjust monthly charges based on services
            if internet_service == 'Fiber optic':
                monthly_charges += np.random.uniform(20, 40)
            elif internet_service == 'DSL':
                monthly_charges += np.random.uniform(10, 25)
            
            if streaming_tv == 'Yes':
                monthly_charges += np.random.uniform(5, 15)
            if streaming_movies == 'Yes':
                monthly_charges += np.random.uniform(5, 15)
            if multiple_lines == 'Yes':
                monthly_charges += np.random.uniform(10, 20)
            
            total_charges = monthly_charges * tenure
            
            # Calculate churn probability based on multiple factors
            churn_prob = 0.2  # Base churn probability
            
            # Tenure factor (newer customers more likely to churn)
            if tenure <= 12:
                churn_prob += 0.3
            elif tenure <= 24:
                churn_prob += 0.15
            elif tenure <= 36:
                churn_prob += 0.05
            
            # Contract type factor
            if contract == 'Month-to-month':
                churn_prob += 0.25
            elif contract == 'One year':
                churn_prob += 0.1
            else:  # Two year
                churn_prob -= 0.05
            
            # Payment method factor
            if payment_method == 'Electronic check':
                churn_prob += 0.15
            elif payment_method == 'Mailed check':
                churn_prob += 0.05
            
            # Service quality factors
            if internet_service == 'Fiber optic':
                if tech_support == 'No':
                    churn_prob += 0.1
                if online_security == 'No':
                    churn_prob += 0.08
            
            # Financial factors
            if monthly_charges > 80:
                churn_prob += 0.1
            elif monthly_charges < 35:
                churn_prob -= 0.05
            
            # Demographic factors
            if senior_citizen == 1:
                churn_prob += 0.05
            
            # Ensure probability is between 0 and 1
            churn_prob = max(0, min(1, churn_prob))
            
            # Final churn decision
            churn = 'Yes' if np.random.random() < churn_prob else 'No'
            
            data.append({
                'customerID': customer_id,
                'gender': gender,
                'SeniorCitizen': senior_citizen,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges,
                'Churn': churn
            })
        
        return pd.DataFrame(data)
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for better model performance"""
        df = df.copy()
        
        # Average monthly charges (total charges / tenure)
        df['AvgMonthlyCharges'] = df['TotalCharges'] / df['tenure'].replace(0, 1)
        
        # Service count (number of additional services)
        service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        def count_services(row):
            count = 0
            for col in service_cols:
                if row[col] == 'Yes':
                    count += 1
            return count
        
        df['ServiceCount'] = df.apply(count_services, axis=1)
        
        # Price per service
        df['PricePerService'] = df['MonthlyCharges'] / (df['ServiceCount'] + 1)
        
        # Tenure categories
        df['TenureGroup'] = pd.cut(df['tenure'], 
                                   bins=[0, 12, 24, 48, 72], 
                                   labels=['0-12', '13-24', '25-48', '49-72'])
        
        # Monthly charges categories
        df['MonthlyChargesGroup'] = pd.cut(df['MonthlyCharges'], 
                                          bins=[0, 35, 65, 95, 120], 
                                          labels=['Low', 'Medium', 'High', 'Premium'])
        
        # Contract length numeric
        contract_mapping = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
        df['ContractNumeric'] = df['Contract'].map(contract_mapping)
        
        # Payment method risk score
        payment_risk = {'Electronic check': 3, 'Mailed check': 2, 'Bank transfer': 1, 'Credit card': 1}
        df['PaymentMethodRisk'] = df['PaymentMethod'].map(payment_risk)
        
        # Senior citizen interaction
        df['SeniorHighCharges'] = (df['SeniorCitizen'] == 1) & (df['MonthlyCharges'] > 80)
        
        # Tenure to charges ratio
        df['TenureToChargesRatio'] = df['tenure'] / df['MonthlyCharges'].replace(0, 1)
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for model training"""
        # Create features
        df = self.create_features(df)
        
        # Separate features and target
        X = df.drop(['Churn', 'customerID'], axis=1)
        y = df['Churn'].map({'Yes': 1, 'No': 0})
        
        # Encode categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col])
            else:
                X[col] = self.label_encoders[col].transform(X[col])
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        return X.values, y.values
    
    def train_model(self, model_type: str = 'xgboost') -> Dict:
        """Train the churn prediction model"""
        # Generate training data
        print("Generating synthetic training data...")
        train_df = self.generate_synthetic_data(5000)
        
        # Preprocess data
        print("Preprocessing data...")
        X, y = self.preprocess_data(train_df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost model
        print(f"Training {model_type} model...")
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print("Model training completed!")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        
        # Store model info
        self.model_info = {
            'model_type': model_type,
            'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'auc_roc': metrics['auc_roc'],
            'training_samples': len(X_train),
            'feature_count': len(self.feature_names)
        }
        
        # Initialize SHAP explainer
        print("Initializing SHAP explainer...")
        try:
            self.shap_explainer = shap.TreeExplainer(self.model)
            print("SHAP explainer initialized successfully!")
        except Exception as e:
            print(f"Warning: Could not initialize SHAP explainer: {e}")
        
        return metrics
    
    def predict(self, customer_data: Dict, include_shap: bool = False) -> Dict:
        """Make churn prediction for a customer"""
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        # Convert input to DataFrame
        df = pd.DataFrame([customer_data])
        
        # Create features
        df = self.create_features(df)
        
        # Encode categorical variables
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        if 'customerID' in categorical_columns:
            categorical_columns.remove('customerID')
        
        for col in categorical_columns:
            if col in self.label_encoders:
                df[col] = self.label_encoders[col].transform(df[col])
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0  # Default value for missing features
        
        # Select features in the correct order
        X = df[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0, 1]
        
        # Determine confidence level
        if probability > 0.8 or probability < 0.2:
            confidence = "High"
        elif probability > 0.6 or probability < 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Generate key factors
        key_factors = []
        if probability > 0.7:
            key_factors.append("High churn probability")
        if customer_data.get('tenure', 0) <= 12:
            key_factors.append("Short tenure")
        if customer_data.get('contract') == 'Month-to-month':
            key_factors.append("Month-to-month contract")
        if customer_data.get('payment_method') == 'Electronic check':
            key_factors.append("Electronic check payment method")
        if customer_data.get('monthly_charges', 0) > 80:
            key_factors.append("High monthly charges")
        
        # Calculate risk score (0-100, higher is riskier)
        risk_score = probability * 100
        
        result = {
            'customer_id': customer_data.get('customer_id', 'Unknown'),
            'churn_probability': float(probability),
            'churn_prediction': 'Yes' if prediction == 1 else 'No',
            'confidence': confidence,
            'key_factors': key_factors[:3],  # Top 3 factors
            'risk_score': float(risk_score)
        }
        
        # Add SHAP explanation if requested and available
        if include_shap and self.shap_explainer is not None:
            try:
                shap_values = self.shap_explainer.shap_values(X_scaled)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # For binary classification
                
                # Create feature importance dictionary
                feature_importance = {}
                for i, feature in enumerate(self.feature_names):
                    feature_importance[feature] = float(shap_values[0][i])
                
                result['shap_explanation'] = {
                    'feature_importance': feature_importance,
                    'base_value': float(self.shap_explainer.expected_value[1] if isinstance(self.shap_explainer.expected_value, list) else self.shap_explainer.expected_value)
                }
            except Exception as e:
                result['shap_explanation'] = {'error': f'Could not generate SHAP explanation: {str(e)}'}
        
        return result
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance from the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
            # Sort by importance
            sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
            return {
                'feature_importance': sorted_importance,
                'top_features': list(sorted_importance.keys())[:15]
            }
        else:
            return {'message': 'Feature importance not available for this model type'}
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'model_info': self.model_info,
            'shap_explainer': self.shap_explainer
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.model_info = model_data['model_info']
        self.shap_explainer = model_data.get('shap_explainer', None)
        print(f"Model loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    # Create and train model
    churn_model = ChurnPredictionModel()
    
    # Train model
    metrics = churn_model.train_model('xgboost')
    
    # Save model
    churn_model.save_model('churn_prediction_model.joblib')
    
    # Test prediction
    test_customer = {
        'customer_id': 'TEST_001',
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 85.5,
        'TotalCharges': 1026.0
    }
    
    prediction = churn_model.predict(test_customer, include_shap=True)
    print("\nTest Prediction:")
    print(f"Customer ID: {prediction['customer_id']}")
    print(f"Churn Prediction: {prediction['churn_prediction']}")
    print(f"Churn Probability: {prediction['churn_probability']:.4f}")
    print(f"Confidence: {prediction['confidence']}")
    print(f"Key Factors: {prediction['key_factors']}")
    print(f"Risk Score: {prediction['risk_score']:.2f}")
    
    if 'shap_explanation' in prediction and 'error' not in prediction['shap_explanation']:
        print(f"SHAP Explanation: {prediction['shap_explanation']['feature_importance']}")
