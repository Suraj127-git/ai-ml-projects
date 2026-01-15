import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PredictiveMaintenanceModel:
    def __init__(self):
        self.rf_model = None
        self.gb_model = None
        self.lr_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.is_trained = False
        
    def generate_synthetic_maintenance_data(self, n_samples=1000):
        """Generate synthetic maintenance data with realistic patterns"""
        np.random.seed(42)
        
        # Equipment types
        equipment_types = ['Pump', 'Motor', 'Compressor', 'Fan', 'Bearing', 'Gearbox']
        
        # Generate base data
        data = []
        
        for i in range(n_samples):
            equipment_type = np.random.choice(equipment_types)
            
            # Base parameters based on equipment type
            if equipment_type == 'Pump':
                base_temp = 45
                base_vibration = 2.5
                base_pressure = 50
                failure_rate = 0.15
            elif equipment_type == 'Motor':
                base_temp = 60
                base_vibration = 3.0
                base_pressure = 30
                failure_rate = 0.20
            elif equipment_type == 'Compressor':
                base_temp = 70
                base_vibration = 4.0
                base_pressure = 80
                failure_rate = 0.25
            elif equipment_type == 'Fan':
                base_temp = 40
                base_vibration = 2.0
                base_pressure = 20
                failure_rate = 0.10
            elif equipment_type == 'Bearing':
                base_temp = 55
                base_vibration = 5.0
                base_pressure = 25
                failure_rate = 0.30
            else:  # Gearbox
                base_temp = 65
                base_vibration = 3.5
                base_pressure = 40
                failure_rate = 0.18
            
            # Generate operational parameters
            operating_hours = np.random.randint(100, 5000)
            load_factor = np.random.uniform(0.5, 1.2)
            
            # Generate sensor readings with some correlation to failure probability
            temperature = base_temp + np.random.normal(0, 5) + (operating_hours * 0.01)
            vibration = base_vibration + np.random.normal(0, 0.5) + (operating_hours * 0.001)
            pressure = base_pressure + np.random.normal(0, 5) + (load_factor * 10)
            
            # Generate maintenance-related features
            days_since_maintenance = np.random.randint(1, 365)
            maintenance_type = np.random.choice(['Preventive', 'Corrective', 'Predictive', 'None'])
            
            # Calculate failure probability based on multiple factors
            failure_prob = failure_rate
            failure_prob += (operating_hours / 10000) * 0.3  # More hours = higher risk
            failure_prob += (days_since_maintenance / 365) * 0.2  # Longer since maintenance = higher risk
            failure_prob += max(0, (temperature - base_temp) / 20) * 0.2  # Higher temp = higher risk
            failure_prob += max(0, (vibration - base_vibration) / 2) * 0.15  # Higher vibration = higher risk
            
            # Add some randomness
            failure_prob += np.random.normal(0, 0.05)
            failure_prob = max(0, min(1, failure_prob))  # Keep between 0 and 1
            
            # Determine failure (binary classification)
            will_fail = 1 if failure_prob > 0.5 else 0
            
            # Generate time to failure (for regression tasks)
            if will_fail:
                time_to_failure = max(1, np.random.randint(1, 30))
            else:
                time_to_failure = np.random.randint(90, 365)
            
            # Create record
            record = {
                'equipment_id': f'EQUIP_{i:04d}',
                'equipment_type': equipment_type,
                'operating_hours': operating_hours,
                'temperature': temperature,
                'vibration': vibration,
                'pressure': pressure,
                'load_factor': load_factor,
                'days_since_maintenance': days_since_maintenance,
                'maintenance_type': maintenance_type,
                'lubrication_level': np.random.uniform(0.3, 1.0),
                'noise_level': np.random.uniform(40, 80),
                'power_consumption': np.random.uniform(80, 120),
                'efficiency': np.random.uniform(0.7, 0.95),
                'age_years': np.random.randint(1, 10),
                'manufacturer': np.random.choice(['A', 'B', 'C', 'D']),
                'installation_date': datetime.now() - timedelta(days=np.random.randint(365, 3650)),
                'will_fail': will_fail,
                'time_to_failure_days': time_to_failure,
                'failure_probability': failure_prob,
                'maintenance_priority': 'High' if failure_prob > 0.7 else 'Medium' if failure_prob > 0.4 else 'Low'
            }
            
            data.append(record)
        
        return pd.DataFrame(data)
    
    def prepare_features(self, df):
        """Prepare features for modeling"""
        df = df.copy()
        
        # Select relevant features
        feature_cols = [
            'operating_hours', 'temperature', 'vibration', 'pressure', 'load_factor',
            'days_since_maintenance', 'lubrication_level', 'noise_level', 
            'power_consumption', 'efficiency', 'age_years'
        ]
        
        categorical_cols = ['equipment_type', 'maintenance_type', 'manufacturer']
        
        # Handle categorical variables
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col])
                else:
                    df[col] = self.label_encoders[col].transform(df[col])
                feature_cols.append(col)
        
        # Add derived features
        df['temp_per_hour'] = df['temperature'] / df['operating_hours']
        df['vibration_per_load'] = df['vibration'] / df['load_factor']
        df['efficiency_degradation'] = 1 - df['efficiency']
        df['high_temp_flag'] = (df['temperature'] > 70).astype(int)
        df['high_vibration_flag'] = (df['vibration'] > 4).astype(int)
        df['overdue_maintenance'] = (df['days_since_maintenance'] > 180).astype(int)
        
        # Update feature columns
        derived_features = ['temp_per_hour', 'vibration_per_load', 'efficiency_degradation', 
                           'high_temp_flag', 'high_vibration_flag', 'overdue_maintenance']
        feature_cols.extend(derived_features)
        
        self.feature_names = feature_cols
        
        return df[feature_cols]
    
    def train_models(self, df):
        """Train all predictive maintenance models"""
        print("Training predictive maintenance models...")
        
        # Prepare features
        X = self.prepare_features(df)
        y_failure = df['will_fail']
        y_time_to_failure = df['time_to_failure_days']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train_failure, y_test_failure = train_test_split(
            X_scaled, y_failure, test_size=0.2, random_state=42, stratify=y_failure
        )
        
        # Train Random Forest for failure prediction
        print("Training Random Forest classifier...")
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        self.rf_model.fit(X_train, y_train_failure)
        
        # Train Gradient Boosting for failure prediction
        print("Training Gradient Boosting classifier...")
        self.gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.gb_model.fit(X_train, y_train_failure)
        
        # Train Logistic Regression for interpretability
        print("Training Logistic Regression classifier...")
        self.lr_model = LogisticRegression(random_state=42, class_weight='balanced')
        self.lr_model.fit(X_train, y_train_failure)
        
        self.is_trained = True
        
        # Evaluate models
        self.evaluate_models(X_test, y_test_failure)
        
        print("All models trained successfully!")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate model performance"""
        print("\\nModel Evaluation Results:\")
        print("=" * 50)
        
        models = {
            'Random Forest': self.rf_model,
            'Gradient Boosting': self.gb_model,
            'Logistic Regression': self.lr_model
        }
        
        results = {}
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            print(f"\\n{name}:")
            print(f"  ROC-AUC: {roc_auc:.4f}")
            print(f"  Classification Report:")
            print(classification_report(y_test, y_pred, target_names=['No Failure', 'Failure']))
            
            results[name] = {
                'roc_auc': roc_auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
        
        return results
    
    def predict_failure_probability(self, equipment_data, model_type='random_forest'):
        """Predict failure probability for equipment"""
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        # Select model
        if model_type == 'random_forest':
            model = self.rf_model
        elif model_type == 'gradient_boosting':
            model = self.gb_model
        elif model_type == 'logistic_regression':
            model = self.lr_model
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Prepare input data
        if isinstance(equipment_data, dict):
            df = pd.DataFrame([equipment_data])
        else:
            df = equipment_data
        
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        failure_probability = model.predict_proba(X_scaled)[:, 1][0]
        will_fail = model.predict(X_scaled)[0]
        
        # Get feature importance (for tree-based models)
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_names, model.feature_importances_))
        
        # Determine maintenance priority
        if failure_probability > 0.8:
            priority = 'Critical'
            recommendation = 'Immediate maintenance required'
        elif failure_probability > 0.6:
            priority = 'High'
            recommendation = 'Schedule maintenance within 1 week'
        elif failure_probability > 0.4:
            priority = 'Medium'
            recommendation = 'Schedule maintenance within 1 month'
        else:
            priority = 'Low'
            recommendation = 'Continue normal operations, monitor closely'
        
        return {
            'equipment_id': equipment_data.get('equipment_id', 'Unknown'),
            'failure_probability': float(failure_probability),
            'will_fail': int(will_fail),
            'maintenance_priority': priority,
            'recommendation': recommendation,
            'model_used': model_type,
            'risk_factors': self.get_risk_factors(equipment_data),
            'feature_importance': feature_importance
        }
    
    def batch_predict_failure_probability(self, equipment_list, model_type='random_forest'):
        """Predict failure probability for multiple equipment"""
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        results = []
        
        for equipment_data in equipment_list:
            try:
                prediction = self.predict_failure_probability(equipment_data, model_type)
                results.append(prediction)
            except Exception as e:
                results.append({
                    'equipment_id': equipment_data.get('equipment_id', 'Unknown'),
                    'error': str(e),
                    'failure_probability': None,
                    'will_fail': None
                })
        
        return results
    
    def get_risk_factors(self, equipment_data):
        """Identify key risk factors for equipment failure"""
        risk_factors = []
        
        # Temperature risk
        if equipment_data.get('temperature', 0) > 70:
            risk_factors.append('High temperature')
        
        # Vibration risk
        if equipment_data.get('vibration', 0) > 4:
            risk_factors.append('High vibration')
        
        # Operating hours risk
        if equipment_data.get('operating_hours', 0) > 3000:
            risk_factors.append('High operating hours')
        
        # Maintenance overdue
        if equipment_data.get('days_since_maintenance', 0) > 180:
            risk_factors.append('Overdue maintenance')
        
        # Low lubrication
        if equipment_data.get('lubrication_level', 1) < 0.5:
            risk_factors.append('Low lubrication level')
        
        # High noise
        if equipment_data.get('noise_level', 0) > 70:
            risk_factors.append('High noise level')
        
        # Low efficiency
        if equipment_data.get('efficiency', 1) < 0.8:
            risk_factors.append('Low efficiency')
        
        # Age risk
        if equipment_data.get('age_years', 0) > 7:
            risk_factors.append('Old equipment')
        
        return risk_factors if risk_factors else ['No major risk factors identified']
    
    def generate_maintenance_schedule(self, equipment_data, model_type='random_forest'):
        """Generate maintenance schedule recommendations"""
        prediction = self.predict_failure_probability(equipment_data, model_type)
        
        failure_prob = prediction['failure_probability']
        
        # Calculate optimal maintenance window
        if failure_prob > 0.8:
            maintenance_window = 'Within 1-3 days'
            estimated_cost = 5000  # High cost for emergency maintenance
        elif failure_prob > 0.6:
            maintenance_window = 'Within 1 week'
            estimated_cost = 3000  # Medium cost for planned maintenance
        elif failure_prob > 0.4:
            maintenance_window = 'Within 2-4 weeks'
            estimated_cost = 2000  # Lower cost for scheduled maintenance
        else:
            maintenance_window = 'Within 3-6 months'
            estimated_cost = 1000  # Minimal cost for routine maintenance
        
        # Calculate potential savings from preventive maintenance
        potential_savings = estimated_cost * 0.3  # Assume 30% savings from preventive vs reactive
        
        return {
            'equipment_id': equipment_data.get('equipment_id', 'Unknown'),
            'failure_probability': failure_prob,
            'maintenance_priority': prediction['maintenance_priority'],
            'recommended_maintenance_window': maintenance_window,
            'estimated_maintenance_cost': estimated_cost,
            'potential_savings_preventive': potential_savings,
            'recommended_maintenance_type': self.get_maintenance_type(failure_prob),
            'critical_components_to_check': self.get_critical_components(equipment_data),
            'downtime_estimate_hours': self.estimate_downtime(failure_prob)
        }
    
    def get_maintenance_type(self, failure_probability):
        """Recommend maintenance type based on failure probability"""
        if failure_probability > 0.8:
            return 'Emergency/Corrective'
        elif failure_probability > 0.6:
            return 'Predictive Maintenance'
        elif failure_probability > 0.4:
            return 'Preventive Maintenance'
        else:
            return 'Routine Inspection'
    
    def get_critical_components(self, equipment_data):
        """Identify critical components to check based on equipment type and condition"""
        equipment_type = equipment_data.get('equipment_type', 'General')
        
        base_components = ['Visual inspection', 'Safety systems', 'Control systems']
        
        if equipment_type == 'Pump':
            specific_components = ['Impeller', 'Seals', 'Bearings', 'Motor coupling']
        elif equipment_type == 'Motor':
            specific_components = ['Bearings', 'Windings', 'Cooling system', 'Coupling']
        elif equipment_type == 'Compressor':
            specific_components = ['Valves', 'Seals', 'Filters', 'Cooling system']
        elif equipment_type == 'Fan':
            specific_components = ['Blades', 'Bearings', 'Belt drive', 'Motor']
        elif equipment_type == 'Bearing':
            specific_components = ['Raceways', 'Rolling elements', 'Lubrication system']
        elif equipment_type == 'Gearbox':
            specific_components = ['Gears', 'Bearings', 'Seals', 'Lubrication system']
        else:
            specific_components = ['Moving parts', 'Lubrication points', 'Wear surfaces']
        
        # Add condition-based components
        if equipment_data.get('temperature', 0) > 70:
            specific_components.append('Cooling system')
        
        if equipment_data.get('vibration', 0) > 4:
            specific_components.append('Alignment/Balance')
        
        if equipment_data.get('days_since_maintenance', 0) > 180:
            specific_components.append('Wear components')
        
        return base_components + specific_components
    
    def estimate_downtime(self, failure_probability):
        """Estimate maintenance downtime in hours"""
        if failure_probability > 0.8:
            return np.random.randint(8, 24)  # Major repair/replacement
        elif failure_probability > 0.6:
            return np.random.randint(4, 8)   # Significant maintenance
        elif failure_probability > 0.4:
            return np.random.randint(2, 4)   # Moderate maintenance
        else:
            return np.random.randint(1, 2)   # Quick inspection/minor maintenance
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'rf_model': self.rf_model,
            'gb_model': self.gb_model,
            'lr_model': self.lr_model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"Predictive maintenance model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        
        self.rf_model = model_data['rf_model']
        self.gb_model = model_data['gb_model']
        self.lr_model = model_data['lr_model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        
        print(f"Predictive maintenance model loaded from {filepath}")

# Training script
if __name__ == "__main__":
    # Initialize model
    maintenance_model = PredictiveMaintenanceModel()
    
    # Generate training data
    print("Generating synthetic maintenance data...")
    df = maintenance_model.generate_synthetic_maintenance_data(n_samples=1000)
    
    print(f"Generated {len(df)} equipment records")
    print(f"Failure rate: {df['will_fail'].mean():.2%}")
    
    # Train models
    print("\\nTraining predictive maintenance models...")
    maintenance_model.train_models(df)
    
    # Save the trained model
    maintenance_model.save_model('predictive_maintenance_model.pkl')
    
    # Test prediction
    print("\\nTesting model prediction...")
    sample_equipment = df.iloc[0].to_dict()
    prediction = maintenance_model.predict_failure_probability(sample_equipment)
    
    print(f\"Equipment ID: {prediction['equipment_id']}\")
    print(f\"Failure Probability: {prediction['failure_probability']:.2%}\")
    print(f\"Maintenance Priority: {prediction['maintenance_priority']}\")
    print(f\"Recommendation: {prediction['recommendation']}\")
    
    print("\\nPredictive maintenance system ready!\")