import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class LeadScoringModel:
    """Lead Scoring Model using multiple ML algorithms"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.is_trained = False
        self.model_performance = {}
        
        # Initialize models
        self.models = {
            'xgboost': None,
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # Initialize scalers
        self.scalers = {
            'standard': StandardScaler()
        }
        
        # Feature importance tracking
        self.feature_importance = {}
        
        # Model thresholds for classification
        self.thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and engineer features for modeling"""
        
        # Create a copy to avoid modifying original data
        features_df = df.copy()
        
        # Encode categorical variables
        categorical_columns = [
            'company_size', 'industry', 'job_title', 'lead_source',
            'budget_range', 'authority_level', 'timeline'
        ]
        
        for col in categorical_columns:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                features_df[col + '_encoded'] = self.encoders[col].fit_transform(features_df[col])
            else:
                features_df[col + '_encoded'] = self.encoders[col].transform(features_df[col])
        
        # Engineer new features
        features_df['engagement_ratio'] = (
            (features_df['email_opens'] + features_df['content_downloads'] + 
             features_df['demo_requests'] * 2 + features_df['form_submissions'] * 1.5) / 
            (features_df['website_visits'] + 1)
        )
        
        features_df['activity_score'] = (
            features_df['website_visits'] * 1 + 
            features_df['email_opens'] * 2 + 
            features_df['form_submissions'] * 3 + 
            features_df['demo_requests'] * 5 + 
            features_df['content_downloads'] * 2 + 
            features_df['social_media_engagement'] * 1.5
        )
        
        features_df['recency_score'] = np.exp(-features_df['days_since_last_activity'] / 30)
        
        # Authority and budget interaction
        features_df['authority_budget_score'] = (
            features_df['authority_level'].map({
                'decision_maker': 4, 'influencer': 3, 'recommender': 2, 'user': 1
            }) * 
            features_df['budget_range'].map({
                'large': 4, 'medium': 3, 'small': 2, 'enterprise': 5
            })
        )
        
        # Timeline urgency
        features_df['timeline_urgency'] = features_df['timeline'].map({
            'immediate': 5, '1_month': 4, '3_months': 3, '6_months': 2, '1_year': 1, 'unknown': 0
        })
        
        # Pain point count and relevance
        features_df['pain_point_count'] = features_df['pain_points'].apply(len)
        
        # Competitor usage impact
        features_df['competitor_factor'] = 1 - features_df['competitor_usage'].astype(int) * 0.2
        
        # Qualification status
        features_df['qualification_score'] = (
            features_df['marketing_qualified'].astype(int) * 2 + 
            features_df['sales_qualified'].astype(int) * 3
        )
        
        # Select final feature set
        feature_columns = [
            'company_size_encoded', 'industry_encoded', 'job_title_encoded', 
            'lead_source_encoded', 'engagement_score', 'website_visits', 'email_opens',
            'form_submissions', 'demo_requests', 'content_downloads', 
            'social_media_engagement', 'days_since_last_activity', 'budget_range_encoded',
            'authority_level_encoded', 'timeline_encoded', 'engagement_ratio',
            'activity_score', 'recency_score', 'authority_budget_score', 
            'timeline_urgency', 'pain_point_count', 'competitor_factor',
            'qualification_score'
        ]
        
        # Store feature names for later use
        self.feature_names = feature_columns
        
        return features_df[feature_columns]
    
    def generate_synthetic_training_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic training data for lead scoring"""
        
        np.random.seed(42)
        
        # Define realistic distributions
        company_sizes = ['small', 'medium', 'large', 'enterprise']
        industries = ['technology', 'healthcare', 'finance', 'retail', 'manufacturing', 'education', 'consulting', 'other']
        job_titles = ['executive', 'manager', 'director', 'vp', 'individual_contributor', 'analyst', 'specialist']
        lead_sources = ['website', 'referral', 'social_media', 'email_campaign', 'event', 'webinar', 'paid_ads', 'organic_search', 'direct']
        budget_ranges = ['small', 'medium', 'large', 'enterprise']
        authority_levels = ['decision_maker', 'influencer', 'recommender', 'user']
        timelines = ['immediate', '1_month', '3_months', '6_months', '1_year', 'unknown']
        pain_points = ['efficiency', 'cost', 'scalability', 'security', 'integration', 'compliance', 'performance', 'support']
        
        data = []
        
        for i in range(n_samples):
            # Generate base characteristics
            company_size = np.random.choice(company_sizes)
            industry = np.random.choice(industries)
            job_title = np.random.choice(job_titles)
            lead_source = np.random.choice(lead_sources)
            budget_range = np.random.choice(budget_ranges)
            authority_level = np.random.choice(authority_levels)
            timeline = np.random.choice(timelines)
            
            # Generate engagement metrics (correlated with conversion)
            base_engagement = np.random.randint(20, 80)
            
            # Higher conversion probability for certain combinations
            conversion_boost = 0
            if company_size in ['large', 'enterprise'] and authority_level in ['decision_maker', 'influencer']:
                conversion_boost += 0.3
            if budget_range in ['large', 'enterprise'] and timeline in ['immediate', '1_month', '3_months']:
                conversion_boost += 0.2
            if industry in ['technology', 'finance'] and lead_source in ['referral', 'website']:
                conversion_boost += 0.15
            
            # Generate activity metrics
            website_visits = max(0, int(np.random.poisson(5 + conversion_boost * 10)))
            email_opens = max(0, int(np.random.poisson(3 + conversion_boost * 7)))
            form_submissions = max(0, int(np.random.poisson(1 + conversion_boost * 3)))
            demo_requests = max(0, int(np.random.poisson(0.5 + conversion_boost * 2)))
            content_downloads = max(0, int(np.random.poisson(2 + conversion_boost * 5)))
            social_media_engagement = max(0, int(np.random.poisson(1 + conversion_boost * 3)))
            
            # Calculate engagement score
            engagement_score = min(100, int(
                website_visits * 2 + email_opens * 3 + form_submissions * 10 + 
                demo_requests * 20 + content_downloads * 5 + social_media_engagement * 3 + 
                base_engagement + conversion_boost * 20
            ))
            
            # Days since last activity (inverse relationship with conversion)
            days_since_last_activity = max(0, int(np.random.exponential(10 - conversion_boost * 5)))
            
            # Pain points
            n_pain_points = max(1, int(np.random.poisson(3)))
            lead_pain_points = np.random.choice(pain_points, size=min(n_pain_points, len(pain_points)), replace=False).tolist()
            
            # Competitor usage (reduces conversion)
            competitor_usage = np.random.random() < (0.3 - conversion_boost * 0.2)
            
            # Marketing and sales qualification
            marketing_qualified = engagement_score > 50 and days_since_last_activity < 30
            sales_qualified = (engagement_score > 70 and 
                              authority_level in ['decision_maker', 'influencer'] and 
                              budget_range in ['large', 'enterprise'] and
                              days_since_last_activity < 14)
            
            # Determine conversion (with some randomness)
            conversion_probability = (
                0.1 +  # Base rate
                conversion_boost +  # Boost from favorable characteristics
                (engagement_score / 100) * 0.4 +  # Engagement impact
                (1 - days_since_last_activity / 100) * 0.2 +  # Recency impact
                (1 - competitor_usage) * 0.1  # Competitor impact
            )
            
            converted = np.random.random() < min(0.9, conversion_probability)
            
            data.append({
                'lead_id': f'LEAD{i+1:04d}',
                'company_size': company_size,
                'industry': industry,
                'job_title': job_title,
                'lead_source': lead_source,
                'engagement_score': engagement_score,
                'website_visits': website_visits,
                'email_opens': email_opens,
                'form_submissions': form_submissions,
                'demo_requests': demo_requests,
                'content_downloads': content_downloads,
                'social_media_engagement': social_media_engagement,
                'days_since_last_activity': days_since_last_activity,
                'budget_range': budget_range,
                'authority_level': authority_level,
                'timeline': timeline,
                'pain_points': lead_pain_points,
                'competitor_usage': competitor_usage,
                'marketing_qualified': marketing_qualified,
                'sales_qualified': sales_qualified,
                'converted': converted
            })
        
        return pd.DataFrame(data)
    
    def train_models(self, df: pd.DataFrame, target_column: str = 'converted') -> Dict[str, Any]:
        """Train all models on the provided data"""
        
        print("🤖 Training lead scoring models...")
        
        # Prepare features
        X = self._prepare_features(df)
        y = df[target_column].astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features for logistic regression
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_test_scaled = self.scalers['standard'].transform(X_test)
        
        # Train models
        results = {}
        
        # XGBoost (if available)
        try:
            import xgboost as xgb
            self.models['xgboost'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
            self.models['xgboost'].fit(X_train, y_train)
            
            # Evaluate XGBoost
            xgb_pred = self.models['xgboost'].predict(X_test)
            xgb_proba = self.models['xgboost'].predict_proba(X_test)[:, 1]
            
            results['xgboost'] = {
                'accuracy': accuracy_score(y_test, xgb_pred),
                'precision': precision_score(y_test, xgb_pred),
                'recall': recall_score(y_test, xgb_pred),
                'f1_score': f1_score(y_test, xgb_pred),
                'auc_roc': roc_auc_score(y_test, xgb_proba),
                'confusion_matrix': confusion_matrix(y_test, xgb_pred).tolist(),
                'predictions': xgb_pred,
                'probabilities': xgb_proba
            }
            
            # Feature importance for XGBoost
            if hasattr(self.models['xgboost'], 'feature_importances_'):
                self.feature_importance['xgboost'] = dict(zip(self.feature_names, self.models['xgboost'].feature_importances_))
            
            print("✅ XGBoost model trained successfully")
            
        except ImportError:
            print("⚠️ XGBoost not available, skipping XGBoost training")
            results['xgboost'] = None
        
        # Random Forest
        self.models['random_forest'].fit(X_train, y_train)
        rf_pred = self.models['random_forest'].predict(X_test)
        rf_proba = self.models['random_forest'].predict_proba(X_test)[:, 1]
        
        results['random_forest'] = {
            'accuracy': accuracy_score(y_test, rf_pred),
            'precision': precision_score(y_test, rf_pred),
            'recall': recall_score(y_test, rf_pred),
            'f1_score': f1_score(y_test, rf_pred),
            'auc_roc': roc_auc_score(y_test, rf_proba),
            'confusion_matrix': confusion_matrix(y_test, rf_pred).tolist(),
            'predictions': rf_pred,
            'probabilities': rf_proba
        }
        
        # Feature importance for Random Forest
        self.feature_importance['random_forest'] = dict(zip(self.feature_names, self.models['random_forest'].feature_importances_))
        
        print("✅ Random Forest model trained successfully")
        
        # Logistic Regression
        self.models['logistic_regression'].fit(X_train_scaled, y_train)
        lr_pred = self.models['logistic_regression'].predict(X_test_scaled)
        lr_proba = self.models['logistic_regression'].predict_proba(X_test_scaled)[:, 1]
        
        results['logistic_regression'] = {
            'accuracy': accuracy_score(y_test, lr_pred),
            'precision': precision_score(y_test, lr_pred),
            'recall': recall_score(y_test, lr_pred),
            'f1_score': f1_score(y_test, lr_pred),
            'auc_roc': roc_auc_score(y_test, lr_proba),
            'confusion_matrix': confusion_matrix(y_test, lr_pred).tolist(),
            'predictions': lr_pred,
            'probabilities': lr_proba
        }
        
        print("✅ Logistic Regression model trained successfully")
        
        # Store test results for later use
        self.model_performance = {
            'test_data': {
                'X_test': X_test,
                'y_test': y_test,
                'X_test_scaled': X_test_scaled
            },
            'results': results,
            'training_date': datetime.now().isoformat(),
            'training_samples': len(X_train)
        }
        
        self.is_trained = True
        
        return results
    
    def predict_lead_score(self, lead_data: Dict[str, Any], model_type: str = 'random_forest') -> Dict[str, Any]:
        """Predict lead score for a single lead"""
        
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        if model_type not in self.models or self.models[model_type] is None:
            raise ValueError(f"Model {model_type} is not available")
        
        # Convert to DataFrame
        df = pd.DataFrame([lead_data])
        
        # Prepare features
        features = self._prepare_features(df)
        
        # Scale features if needed
        if model_type == 'logistic_regression':
            features_scaled = self.scalers['standard'].transform(features)
            prediction_features = features_scaled
        else:
            prediction_features = features
        
        # Make prediction
        model = self.models[model_type]
        probability = model.predict_proba(prediction_features)[0, 1]
        prediction = model.predict(prediction_features)[0]
        
        # Calculate score (0-100)
        score = int(probability * 100)
        
        # Generate recommendation
        if score >= self.thresholds['high'] * 100:
            recommendation = "HIGH PRIORITY - Immediate sales follow-up recommended"
        elif score >= self.thresholds['medium'] * 100:
            recommendation = "MEDIUM PRIORITY - Nurture and follow up within 1 week"
        elif score >= self.thresholds['low'] * 100:
            recommendation = "LOW PRIORITY - Continue nurturing with marketing campaigns"
        else:
            recommendation = "COLD LEAD - Focus on awareness and education"
        
        # Get feature importance for this prediction
        if model_type in self.feature_importance:
            factors = self.feature_importance[model_type]
        else:
            factors = {}
        
        return {
            'score': score,
            'conversion_probability': float(probability),
            'model_used': model_type,
            'confidence': float(probability * 0.9 + 0.1),  # Simple confidence metric
            'recommendation': recommendation,
            'factors': factors,
            'prediction': int(prediction)
        }
    
    def batch_predict(self, leads_data: List[Dict[str, Any]], model_type: str = 'random_forest') -> List[Dict[str, Any]]:
        """Predict lead scores for multiple leads"""
        
        results = []
        
        for lead_data in leads_data:
            try:
                prediction = self.predict_lead_score(lead_data, model_type)
                prediction['lead_id'] = lead_data.get('lead_id', 'unknown')
                results.append(prediction)
            except Exception as e:
                # Handle individual lead prediction failures
                results.append({
                    'lead_id': lead_data.get('lead_id', 'unknown'),
                    'score': 0,
                    'conversion_probability': 0.0,
                    'model_used': model_type,
                    'confidence': 0.0,
                    'recommendation': f'Prediction failed: {str(e)}',
                    'factors': {},
                    'prediction': 0
                })
        
        return results
    
    def get_model_performance(self, model_type: str = None) -> Dict[str, Any]:
        """Get model performance metrics"""
        
        if not self.is_trained:
            return {'error': 'Models not trained yet'}
        
        if model_type:
            if model_type in self.model_performance['results']:
                return self.model_performance['results'][model_type]
            else:
                return {'error': f'Model {model_type} not found'}
        
        return self.model_performance['results']
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get general model information"""
        
        return {
            'model_name': 'Lead Scoring Model',
            'version': '1.0.0',
            'description': 'ML-powered lead scoring using XGBoost, Random Forest, and Logistic Regression',
            'features': self.feature_names,
            'algorithms': list(self.models.keys()),
            'training_date': self.model_performance.get('training_date'),
            'last_updated': datetime.now().isoformat(),
            'model_status': 'trained' if self.is_trained else 'not_trained',
            'training_samples': self.model_performance.get('training_samples', 0),
            'thresholds': self.thresholds
        }
    
    def save_models(self, filepath: str):
        """Save trained models to file"""
        
        if not self.is_trained:
            raise ValueError("Models must be trained before saving")
        
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
            'model_performance': self.model_performance,
            'feature_importance': self.feature_importance,
            'thresholds': self.thresholds
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models from file"""
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.encoders = model_data['encoders']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        self.model_performance = model_data['model_performance']
        self.feature_importance = model_data.get('feature_importance', {})
        self.thresholds = model_data.get('thresholds', self.thresholds)
        
        print(f"✅ Models loaded from {filepath}")
    
    def generate_synthetic_lead_data(self, n_leads: int = 10) -> List[Dict[str, Any]]:
        """Generate synthetic lead data for testing"""
        
        # Use the same logic as training data generation but without conversion labels
        df = self.generate_synthetic_training_data(n_samples=n_leads)
        
        # Remove the conversion column and convert to list of dicts
        df = df.drop('converted', axis=1)
        
        return df.to_dict('records')