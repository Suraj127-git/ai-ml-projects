import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from sklearn.ensemble import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta

class EnergyConsumptionModel:
    """Energy Consumption Forecasting Model with XGBoost"""
    
    def __init__(self):
        self.xgboost_model = None
        self.feature_scaler = StandardScaler()
        self.is_fitted = False
        
    def generate_synthetic_energy_data(self, n_days: int = 365, building_type: str = "office") -> pd.DataFrame:
        """Generate synthetic energy consumption data"""
        
        np.random.seed(42)
        start_date = datetime.now() - timedelta(days=n_days)
        dates = pd.date_range(start=start_date, periods=n_days*24, freq='H')
        
        building_profiles = {
            "office": {"base_load": 50, "peak_multiplier": 2.5, "peak_hours": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]},
            "residential": {"base_load": 30, "peak_multiplier": 2.0, "peak_hours": [6, 7, 8, 18, 19, 20, 21, 22]},
            "retail": {"base_load": 40, "peak_multiplier": 3.0, "peak_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]},
            "industrial": {"base_load": 100, "peak_multiplier": 1.8, "peak_hours": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]},
            "hospital": {"base_load": 80, "peak_multiplier": 1.5, "peak_hours": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]},
            "school": {"base_load": 35, "peak_multiplier": 2.8, "peak_hours": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]}
        }
        
        profile = building_profiles.get(building_type, building_profiles["office"])
        energy_consumption = []
        
        for i, date in enumerate(dates):
            hour = date.hour
            day_of_week = date.dayofweek
            month = date.month
            
            base_consumption = profile["base_load"]
            
            if hour in profile["peak_hours"]:
                hourly_factor = profile["peak_multiplier"]
            elif 0 <= hour <= 5:
                hourly_factor = 0.3
            else:
                hourly_factor = 0.8
            
            weekend_factor = 0.6 if day_of_week >= 5 and building_type == "office" else 1.0
            
            if month in [12, 1, 2]:
                seasonal_factor = 1.3
            elif month in [6, 7, 8]:
                seasonal_factor = 1.2
            else:
                seasonal_factor = 1.0
            
            temperature = np.random.normal(20, 8)
            temp_effect = abs(temperature - 20) * 0.02
            
            consumption = (base_consumption * hourly_factor * weekend_factor * seasonal_factor * 
                          (1 + temp_effect) * np.random.normal(1, 0.1))
            
            energy_consumption.append(max(0, consumption))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'energy_consumption_kwh': energy_consumption,
            'temperature_celsius': [np.random.normal(20, 8) for _ in range(len(dates))],
            'humidity_percent': [np.random.normal(60, 15) for _ in range(len(dates))],
            'occupancy_rate': [np.random.uniform(0.3, 0.9) for _ in range(len(dates))],
            'weather_condition': [np.random.choice(['sunny', 'cloudy', 'rainy', 'snowy']) for _ in range(len(dates))],
            'is_holiday': [np.random.choice([True, False], p=[0.05, 0.95]) for _ in range(len(dates))],
            'building_type': building_type
        })
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for machine learning models"""
        
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        weather_mapping = {'sunny': 0, 'cloudy': 1, 'rainy': 2, 'snowy': 3}
        df['weather_encoded'] = df['weather_condition'].map(weather_mapping)
        
        df['energy_lag_1h'] = df['energy_consumption_kwh'].shift(1)
        df['energy_lag_24h'] = df['energy_consumption_kwh'].shift(24)
        df['energy_lag_7d'] = df['energy_consumption_kwh'].shift(24*7)
        
        df['energy_rolling_mean_24h'] = df['energy_consumption_kwh'].rolling(window=24, min_periods=1).mean()
        df['energy_rolling_std_24h'] = df['energy_consumption_kwh'].rolling(window=24, min_periods=1).std()
        
        df = df.fillna(df.mean())
        
        return df
    
    def train_xgboost_model(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train XGBoost model for energy consumption forecasting"""
        
        df_prepared = self.prepare_features(df)
        
        feature_cols = ['hour', 'day_of_week', 'month', 'is_weekend', 'hour_sin', 'hour_cos',
                         'day_sin', 'day_cos', 'month_sin', 'month_cos', 'temperature_celsius',
                         'humidity_percent', 'occupancy_rate', 'weather_encoded', 'is_holiday',
                         'energy_lag_1h', 'energy_lag_24h', 'energy_lag_7d', 'energy_rolling_mean_24h',
                         'energy_rolling_std_24h']
        
        X = df_prepared[feature_cols]
        y = df_prepared['energy_consumption_kwh']
        
        split_idx = int(0.8 * len(df))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        self.xgboost_model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        self.xgboost_model.fit(X_train_scaled, y_train)
        
        y_pred = self.xgboost_model.predict(X_test_scaled)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        self.is_fitted = True
        return metrics
    
    def forecast_xgboost(self, historical_df: pd.DataFrame, forecast_hours: int) -> Tuple[List[Dict], Dict]:
        """Generate forecast using XGBoost model"""
        
        if self.xgboost_model is None:
            raise ValueError("XGBoost model not trained. Train the model first.")
        
        df_prepared = self.prepare_features(historical_df)
        
        feature_cols = ['hour', 'day_of_week', 'month', 'is_weekend', 'hour_sin', 'hour_cos',
                         'day_sin', 'day_cos', 'month_sin', 'month_cos', 'temperature_celsius',
                         'humidity_percent', 'occupancy_rate', 'weather_encoded', 'is_holiday',
                         'energy_lag_1h', 'energy_lag_24h', 'energy_lag_7d', 'energy_rolling_mean_24h',
                         'energy_rolling_std_24h']
        
        last_timestamp = pd.to_datetime(historical_df['timestamp'].iloc[-1])
        future_timestamps = pd.date_range(start=last_timestamp + timedelta(hours=1), 
                                        periods=forecast_hours, freq='H')
        
        future_data = []
        for timestamp in future_timestamps:
            avg_temp = historical_df['temperature_celsius'].mean()
            avg_humidity = historical_df['humidity_percent'].mean()
            avg_occupancy = historical_df['occupancy_rate'].mean()
            
            weather_counts = historical_df['weather_condition'].value_counts()
            weather_prob = weather_counts / weather_counts.sum()
            weather_condition = np.random.choice(weather_counts.index, p=weather_prob.values)
            
            future_data.append({
                'timestamp': timestamp,
                'temperature_celsius': avg_temp + np.random.normal(0, 2),
                'humidity_percent': avg_humidity + np.random.normal(0, 5),
                'occupancy_rate': avg_occupancy + np.random.normal(0, 0.1),
                'weather_condition': weather_condition,
                'is_holiday': False,
                'building_type': historical_df['building_type'].iloc[0] if 'building_type' in historical_df.columns else 'office'
            })
        
        future_df = pd.DataFrame(future_data)
        future_prepared = self.prepare_features(pd.concat([historical_df, future_df], ignore_index=True))
        future_features = future_prepared[feature_cols].tail(forecast_hours)
        
        future_features_scaled = self.feature_scaler.transform(future_features)
        
        forecast_values = self.xgboost_model.predict(future_features_scaled)
        
        forecast_data = []
        for i, (timestamp, value) in enumerate(zip(future_timestamps, forecast_values)):
            forecast_data.append({
                'timestamp': timestamp.isoformat(),
                'energy_consumption_kwh': float(max(0, value)),
                'confidence_lower': float(max(0, value * 0.9)),
                'confidence_upper': float(max(0, value * 1.1))
            })
        
        total_consumption = sum(item['energy_consumption_kwh'] for item in forecast_data)
        avg_consumption = total_consumption / len(forecast_data)
        
        peak_hour_data = max(forecast_data, key=lambda x: x['energy_consumption_kwh'])
        
        metadata = {
            'total_forecast_consumption': total_consumption,
            'average_hourly_consumption': avg_consumption,
            'peak_consumption_hour': peak_hour_data['timestamp'],
            'peak_consumption_value': peak_hour_data['energy_consumption_kwh']
        }
        
        return forecast_data, metadata