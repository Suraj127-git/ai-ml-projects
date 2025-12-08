import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import joblib
import os
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class LoanEligibilityModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.model_info = {}
        
    def generate_synthetic_data(self, n_samples: int = 5000) -> pd.DataFrame:
        """Generate synthetic loan application data"""
        np.random.seed(42)
        
        data = []
        for _ in range(n_samples):
            # Generate basic demographics
            gender = np.random.choice(['Male', 'Female'])
            married = np.random.choice(['Yes', 'No'])
            dependents = np.random.randint(0, 4)
            education = np.random.choice(['Graduate', 'Not Graduate'])
            self_employed = np.random.choice(['Yes', 'No'])
            
            # Generate income and loan details
            applicant_income = np.random.lognormal(10.5, 0.5)  # Log-normal distribution
            coapplicant_income = np.random.lognormal(9.5, 0.8) if np.random.random() > 0.3 else 0
            loan_amount = np.random.lognormal(12, 0.3)
            loan_amount_term = np.random.choice([12, 24, 36, 60, 84, 120, 180, 240, 300, 360, 480])
            
            # Credit history (biased towards good credit)
            credit_history = np.random.choice(['Yes', 'No'], p=[0.85, 0.15])
            
            # Property area
            property_area = np.random.choice(['Urban', 'Semiurban', 'Rural'], p=[0.5, 0.3, 0.2])
            
            # Calculate eligibility based on multiple factors
            # Base probability
            approval_prob = 0.6
            
            # Income factors
            total_income = applicant_income + coapplicant_income
            income_to_loan_ratio = total_income / loan_amount
            if income_to_loan_ratio > 0.3:
                approval_prob += 0.2
            elif income_to_loan_ratio > 0.2:
                approval_prob += 0.1
            else:
                approval_prob -= 0.3
            
            # Credit history factor
            if credit_history == 'Yes':
                approval_prob += 0.3
            else:
                approval_prob -= 0.4
            
            # Education factor
            if education == 'Graduate':
                approval_prob += 0.1
            else:
                approval_prob -= 0.1
            
            # Employment factor
            if self_employed == 'No':
                approval_prob += 0.05
            else:
                approval_prob -= 0.1
            
            # Property area factor
            if property_area == 'Urban':
                approval_prob += 0.05
            elif property_area == 'Semiurban':
                approval_prob += 0.02
            else:
                approval_prob -= 0.05
            
            # Marital status factor
            if married == 'Yes':
                approval_prob += 0.05
            
            # Dependents factor
            if dependents == 0:
                approval_prob += 0.05
            elif dependents <= 2:
                approval_prob -= 0.02
            else:
                approval_prob -= 0.1
            
            # Loan term factor
            if loan_amount_term <= 60:
                approval_prob += 0.05
            elif loan_amount_term <= 180:
                approval_prob += 0.02
            else:
                approval_prob -= 0.05
            
            # Ensure probability is between 0 and 1
            approval_prob = max(0, min(1, approval_prob))
            
            # Final decision based on probability
            loan_status = 'Y' if np.random.random() < approval_prob else 'N'
            
            data.append({
                'Gender': gender,
                'Married': married,
                'Dependents': dependents,
                'Education': education,
                'Self_Employed': self_employed,
                'ApplicantIncome': applicant_income,
                'CoapplicantIncome': coapplicant_income,
                'LoanAmount': loan_amount,
                'Loan_Amount_Term': loan_amount_term,
                'Credit_History': credit_history,
                'Property_Area': property_area,
                'Loan_Status': loan_status
            })
        
        return pd.DataFrame(data)
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for better model performance"""
        df = df.copy()
        
        # Total income
        df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        
        # Income to loan ratio
        df['Income_Loan_Ratio'] = df['Total_Income'] / df['LoanAmount']
        
        # Per capita income (income per dependent)
        df['Per_Capita_Income'] = df['Total_Income'] / (df['Dependents'] + 1)
        
        # Loan amount per month
        df['Monthly_Loan_Amount'] = df['LoanAmount'] / df['Loan_Amount_Term']
        
        # Income stability indicator
        df['Income_Stability'] = (df['Self_Employed'] == 'No').astype(int)
        
        # Family size
        df['Family_Size'] = df['Dependents'] + (df['Married'] == 'Yes').astype(int) + 1
        
        # Credit history binary
        df['Credit_History_Binary'] = (df['Credit_History'] == 'Yes').astype(int)
        
        # Property area encoded numerically
        property_area_mapping = {'Rural': 0, 'Semiurban': 1, 'Urban': 2}
        df['Property_Area_Numeric'] = df['Property_Area'].map(property_area_mapping)
        
        # Education level numeric
        df['Education_Numeric'] = (df['Education'] == 'Graduate').astype(int)
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for model training"""
        # Create features
        df = self.create_features(df)
        
        # Separate features and target
        X = df.drop(['Loan_Status'], axis=1)
        y = df['Loan_Status'].map({'Y': 1, 'N': 0})
        
        # Encode categorical variables
        categorical_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col])
            else:
                X[col] = self.label_encoders[col].transform(X[col])
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        return X.values, y.values
    
    def train_model(self, model_type: str = 'gradient_boosting') -> Dict:
        """Train the loan eligibility model"""
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
        
        # Train model
        print(f"Training {model_type} model...")
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        else:  # gradient_boosting
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
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
            'training_samples': len(X_train)
        }
        
        return metrics
    
    def predict(self, applicant_data: Dict) -> Dict:
        """Make loan eligibility prediction"""
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        # Convert input to DataFrame
        df = pd.DataFrame([applicant_data])
        
        # Create features
        df = self.create_features(df)
        
        # Encode categorical variables
        categorical_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
        
        for col in categorical_columns:
            if col in df.columns:
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
        if applicant_data.get('credit_history', '').lower() == 'yes':
            key_factors.append("Good credit history")
        else:
            key_factors.append("Poor credit history")
        
        total_income = applicant_data.get('applicant_income', 0) + applicant_data.get('coapplicant_income', 0)
        loan_amount = applicant_data.get('loan_amount', 0)
        if loan_amount > 0:
            income_ratio = total_income / loan_amount
            if income_ratio > 0.3:
                key_factors.append("Strong income-to-loan ratio")
            elif income_ratio > 0.2:
                key_factors.append("Moderate income-to-loan ratio")
            else:
                key_factors.append("Weak income-to-loan ratio")
        
        if applicant_data.get('education', '').lower() == 'graduate':
            key_factors.append("Graduate education level")
        
        if applicant_data.get('property_area', '').lower() == 'urban':
            key_factors.append("Urban property location")
        
        # Calculate risk score (0-100, higher is riskier)
        risk_score = (1 - probability) * 100
        
        return {
            'eligibility': 'Approved' if prediction == 1 else 'Rejected',
            'probability': float(probability),
            'confidence': confidence,
            'key_factors': key_factors[:3],  # Top 3 factors
            'risk_score': float(risk_score)
        }
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'model_info': self.model_info
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
        print(f"Model loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    # Create and train model
    loan_model = LoanEligibilityModel()
    
    # Train with gradient boosting (default)
    metrics = loan_model.train_model('gradient_boosting')
    
    # Save model
    loan_model.save_model('loan_eligibility_model.joblib')
    
    # Test prediction
    test_applicant = {
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': 1,
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 5000,
        'CoapplicantIncome': 3000,
        'LoanAmount': 150000,
        'Loan_Amount_Term': 360,
        'Credit_History': 'Yes',
        'Property_Area': 'Urban'
    }
    
    prediction = loan_model.predict(test_applicant)
    print("\nTest Prediction:")
    print(f"Eligibility: {prediction['eligibility']}")
    print(f"Probability: {prediction['probability']:.4f}")
    print(f"Confidence: {prediction['confidence']}")
    print(f"Key Factors: {prediction['key_factors']}")
    print(f"Risk Score: {prediction['risk_score']:.2f}")