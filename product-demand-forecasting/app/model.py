import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

try:
    import prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

class DemandForecastingModel:
    """
    Product Demand Forecasting Model
    Supports ARIMA, Prophet, LSTM, and XGBoost models
    """
    
    def __init__(self):
        self.arima_models = {}
        self.prophet_models = {}
        self.lstm_models = {}
        self.xgboost_models = {}
        self.scalers = {}
        self.model_performance = {}
        
    def generate_synthetic_demand_data(self, n_products: int = 10, n_days: int = 730) -> pd.DataFrame:
        """Generate synthetic product demand data"""
        np.random.seed(42)
        
        products = []
        start_date = datetime.now() - timedelta(days=n_days)
        
        for product_id in range(1, n_products + 1):
            product_name = f"PRODUCT_{product_id:03d}"
            base_demand = np.random.uniform(50, 200)
            trend = np.random.uniform(-0.01, 0.03)
            seasonality_strength = np.random.uniform(0.1, 0.5)
            
            for day in range(n_days):
                current_date = start_date + timedelta(days=day)
                
                # Trend component
                trend_component = base_demand + (trend * day)
                
                # Seasonal component (weekly and monthly patterns)
                weekly_season = 10 * np.sin(2 * np.pi * day / 7)
                monthly_season = 20 * np.sin(2 * np.pi * day / 30)
                seasonal_component = seasonality_strength * (weekly_season + monthly_season)
                
                # Random noise
                noise = np.random.normal(0, base_demand * 0.1)
                
                # Demand calculation
                demand = max(0, trend_component + seasonal_component + noise)
                
                # Additional features
                price = np.random.uniform(10, 100)
                promotion = np.random.choice([0, 1], p=[0.8, 0.2])
                holiday = 1 if current_date.weekday() >= 5 else 0  # Weekend effect
                stock_level = np.random.randint(50, 500)
                
                products.append({
                    'product_id': product_name,
                    'date': current_date,
                    'demand': demand,
                    'price': price,
                    'promotion': promotion,
                    'seasonality': seasonal_component,
                    'holiday': holiday,
                    'stock_level': stock_level
                })
        
        return pd.DataFrame(products)
    
    def prepare_time_series_data(self, df: pd.DataFrame, product_id: str) -> pd.DataFrame:
        """Prepare time series data for a specific product"""
        product_data = df[df['product_id'] == product_id].copy()
        product_data = product_data.sort_values('date')
        product_data.set_index('date', inplace=True)
        
        # Add time-based features
        product_data['day_of_week'] = product_data.index.dayofweek
        product_data['month'] = product_data.index.month
        product_data['quarter'] = product_data.index.quarter
        product_data['year'] = product_data.index.year
        product_data['day_of_year'] = product_data.index.dayofyear
        
        # Add lag features
        product_data['demand_lag_1'] = product_data['demand'].shift(1)
        product_data['demand_lag_7'] = product_data['demand'].shift(7)
        product_data['demand_lag_30'] = product_data['demand'].shift(30)
        
        # Add rolling statistics
        product_data['demand_ma_7'] = product_data['demand'].rolling(window=7).mean()
        product_data['demand_ma_30'] = product_data['demand'].rolling(window=30).mean()
        product_data['demand_std_7'] = product_data['demand'].rolling(window=7).std()
        
        return product_data.dropna()
    
    def check_stationarity(self, series: pd.Series) -> Tuple[bool, float]:
        """Check if time series is stationary using Augmented Dickey-Fuller test"""
        result = adfuller(series.dropna())
        is_stationary = result[1] <= 0.05
        p_value = result[1]
        return is_stationary, p_value
    
    def train_arima_model(self, df: pd.DataFrame, product_id: str) -> Dict[str, Any]:
        """Train ARIMA model for demand forecasting"""
        print(f"Training ARIMA model for {product_id}...")
        
        # Prepare data
        product_data = self.prepare_time_series_data(df, product_id)
        demand_series = product_data['demand']
        
        # Check stationarity
        is_stationary, p_value = self.check_stationarity(demand_series)
        
        if not is_stationary:
            # Apply differencing
            demand_diff = demand_series.diff().dropna()
        else:
            demand_diff = demand_series
        
        # Split data
        train_size = int(len(demand_diff) * 0.8)
        train_data = demand_diff[:train_size]
        test_data = demand_diff[train_size:]
        
        # Train ARIMA model
        try:
            model = ARIMA(train_data, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # Make predictions
            predictions = fitted_model.forecast(steps=len(test_data))
            
            # Calculate metrics
            mae = mean_absolute_error(test_data, predictions)
            rmse = np.sqrt(mean_squared_error(test_data, predictions))
            mape = mean_absolute_percentage_error(test_data, predictions) * 100
            
            # Store model
            self.arima_models[product_id] = fitted_model
            
            performance = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'is_stationary': is_stationary,
                'p_value': p_value,
                'training_samples': len(train_data),
                'test_samples': len(test_data)
            }
            
            self.model_performance[f'{product_id}_arima'] = performance
            print(f"ARIMA model trained! RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
            
            return performance
            
        except Exception as e:
            print(f"ARIMA training failed: {e}")
            return {'error': str(e)}
    
    def train_prophet_model(self, df: pd.DataFrame, product_id: str) -> Dict[str, Any]:
        """Train Prophet model for demand forecasting"""
        if not PROPHET_AVAILABLE:
            raise RuntimeError("Prophet not available. Please install it: pip install prophet")
        
        print(f"Training Prophet model for {product_id}...")
        
        # Prepare data
        product_data = self.prepare_time_series_data(df, product_id)
        
        # Prepare Prophet format
        prophet_df = pd.DataFrame({
            'ds': product_data.index,
            'y': product_data['demand']
        })
        
        # Add regressors
        prophet_df['promotion'] = product_data['promotion']
        prophet_df['holiday'] = product_data['holiday']
        prophet_df['price'] = product_data['price']
        
        # Split data
        train_size = int(len(prophet_df) * 0.8)
        train_df = prophet_df[:train_size]
        test_df = prophet_df[train_size:]
        
        # Train Prophet model
        model = prophet.Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        
        # Add regressors
        model.add_regressor('promotion')
        model.add_regressor('holiday')
        model.add_regressor('price')
        
        model.fit(train_df)
        
        # Make predictions
        future = model.make_future_dataframe(periods=len(test_df), freq='D')
        future['promotion'] = prophet_df['promotion'].values[-len(future):]
        future['holiday'] = prophet_df['holiday'].values[-len(future):]
        future['price'] = prophet_df['price'].values[-len(future):]
        
        forecast = model.predict(future)
        predictions = forecast['yhat'].values[-len(test_df):]
        
        # Calculate metrics
        mae = mean_absolute_error(test_df['y'], predictions)
        rmse = np.sqrt(mean_squared_error(test_df['y'], predictions))
        mape = mean_absolute_percentage_error(test_df['y'], predictions) * 100
        
        # Store model
        self.prophet_models[product_id] = model
        
        performance = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'training_samples': len(train_df),
            'test_samples': len(test_df)
        }
        
        self.model_performance[f'{product_id}_prophet'] = performance
        print(f"Prophet model trained! RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        
        return performance
    
    def train_lstm_model(self, df: pd.DataFrame, product_id: str) -> Dict[str, Any]:
        """Train LSTM model for demand forecasting"""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow not available. Please install it: pip install tensorflow")
        
        print(f"Training LSTM model for {product_id}...")
        
        # Prepare data
        product_data = self.prepare_time_series_data(df, product_id)
        
        # Select features
        feature_cols = ['demand', 'price', 'promotion', 'holiday', 'day_of_week', 'month', 'demand_lag_1', 'demand_lag_7']
        X = product_data[feature_cols].values
        y = product_data['demand'].values
        
        # Scale data
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Create sequences
        def create_sequences(data, target, seq_length):
            X_seq, y_seq = [], []
            for i in range(seq_length, len(data)):
                X_seq.append(data[i-seq_length:i])
                y_seq.append(target[i])
            return np.array(X_seq), np.array(y_seq)
        
        seq_length = 30  # Use 30 days of history
        X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)
        
        # Split data
        train_size = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:train_size], X_seq[train_size:]
        y_train, y_test = y_seq[:train_size], y_seq[train_size:]
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(seq_length, len(feature_cols))),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Make predictions
        predictions_scaled = model.predict(X_test)
        predictions = scaler_y.inverse_transform(predictions_scaled).flatten()
        y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(y_test_actual, predictions)
        rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
        mape = mean_absolute_percentage_error(y_test_actual, predictions) * 100
        
        # Store model and scalers
        self.lstm_models[product_id] = model
        self.scalers[f'{product_id}_X'] = scaler_X
        self.scalers[f'{product_id}_y'] = scaler_y
        
        performance = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'epochs_trained': len(history.history['loss'])
        }
        
        self.model_performance[f'{product_id}_lstm'] = performance
        print(f"LSTM model trained! RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        
        return performance
    
    def train_xgboost_model(self, df: pd.DataFrame, product_id: str) -> Dict[str, Any]:
        """Train XGBoost model for demand forecasting"""
        if not XGBOOST_AVAILABLE:
            raise RuntimeError("XGBoost not available. Please install it: pip install xgboost")
        
        print(f"Training XGBoost model for {product_id}...")
        
        # Prepare data
        product_data = self.prepare_time_series_data(df, product_id)
        
        # Select features
        feature_cols = ['price', 'promotion', 'holiday', 'day_of_week', 'month', 
                       'demand_lag_1', 'demand_lag_7', 'demand_ma_7', 'demand_ma_30']
        
        X = product_data[feature_cols].values
        y = product_data['demand'].values
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            objective='reg:squarederror'
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mape = mean_absolute_percentage_error(y_test, predictions) * 100
        
        # Store model
        self.xgboost_models[product_id] = model
        
        performance = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        self.model_performance[f'{product_id}_xgboost'] = performance
        print(f"XGBoost model trained! RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        
        return performance
    
    def forecast_demand(self, product_id: str, model_type: str, periods: int = 30, 
                       historical_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Generate demand forecast for a product"""
        
        if model_type == 'arima':
            return self._forecast_arima(product_id, periods, historical_data)
        elif model_type == 'prophet':
            return self._forecast_prophet(product_id, periods, historical_data)
        elif model_type == 'lstm':
            return self._forecast_lstm(product_id, periods, historical_data)
        elif model_type == 'xgboost':
            return self._forecast_xgboost(product_id, periods, historical_data)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _forecast_arima(self, product_id: str, periods: int, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate ARIMA forecast"""
        if product_id not in self.arima_models:
            raise RuntimeError(f"ARIMA model not trained for {product_id}")
        
        model = self.arima_models[product_id]
        
        # Generate forecast
        forecast = model.forecast(steps=periods)
        confidence_intervals = model.get_forecast(steps=periods).conf_int()
        
        # Create forecast dates
        last_date = historical_data['date'].max() if historical_data is not None else datetime.now()
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(periods)]
        
        forecast_data = []
        for i, (date, pred) in enumerate(zip(forecast_dates, forecast)):
            forecast_data.append({
                'date': date.isoformat(),
                'demand': max(0, float(pred)),  # Ensure non-negative demand
                'lower_bound': max(0, float(confidence_intervals.iloc[i, 0])),
                'upper_bound': max(0, float(confidence_intervals.iloc[i, 1]))
            })
        
        return {
            'forecast': forecast_data,
            'model_type': 'arima',
            'forecast_periods': periods,
            'confidence_intervals': {
                'lower': [f['lower_bound'] for f in forecast_data],
                'upper': [f['upper_bound'] for f in forecast_data]
            }
        }
    
    def _forecast_prophet(self, product_id: str, periods: int, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate Prophet forecast"""
        if product_id not in self.prophet_models:
            raise RuntimeError(f"Prophet model not trained for {product_id}")
        
        model = self.prophet_models[product_id]
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods, freq='D')
        
        # Add regressors (use last values as baseline)
        if historical_data is not None:
            last_values = historical_data[historical_data['product_id'] == product_id].tail(1)
            if not last_values.empty:
                future['promotion'] = last_values['promotion'].iloc[0]
                future['holiday'] = last_values['holiday'].iloc[0]
                future['price'] = last_values['price'].iloc[0]
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Extract last 'periods' predictions
        forecast_tail = forecast.tail(periods)
        
        forecast_data = []
        for _, row in forecast_tail.iterrows():
            forecast_data.append({
                'date': row['ds'].isoformat(),
                'demand': max(0, float(row['yhat'])),
                'lower_bound': max(0, float(row['yhat_lower'])),
                'upper_bound': max(0, float(row['yhat_upper']))
            })
        
        return {
            'forecast': forecast_data,
            'model_type': 'prophet',
            'forecast_periods': periods,
            'confidence_intervals': {
                'lower': [f['lower_bound'] for f in forecast_data],
                'upper': [f['upper_bound'] for f in forecast_data]
            }
        }
    
    def save_model(self, product_id: str, model_type: str, filepath: str):
        """Save trained model"""
        model_data = {
            'product_id': product_id,
            'model_type': model_type,
            'models': {
                'arima': self.arima_models.get(product_id),
                'prophet': self.prophet_models.get(product_id),
                'lstm': self.lstm_models.get(product_id),
                'xgboost': self.xgboost_models.get(product_id)
            },
            'scalers': self.scalers,
            'model_performance': self.model_performance,
            'saved_at': datetime.now().isoformat()
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        model_data = joblib.load(filepath)
        
        # Restore models
        product_id = model_data['product_id']
        models = model_data['models']
        
        if models['arima']:
            self.arima_models[product_id] = models['arima']
        if models['prophet']:
            self.prophet_models[product_id] = models['prophet']
        if models['lstm']:
            self.lstm_models[product_id] = models['lstm']
        if models['xgboost']:
            self.xgboost_models[product_id] = models['xgboost']
        
        self.scalers = model_data.get('scalers', {})
        self.model_performance = model_data.get('model_performance', {})
        
        print(f"Model loaded from {filepath}")
    
    def get_model_info(self, product_id: str) -> Dict[str, Any]:
        """Get information about trained models for a product"""
        info = {
            'product_id': product_id,
            'models_available': {
                'arima': product_id in self.arima_models,
                'prophet': product_id in self.prophet_models,
                'lstm': product_id in self.lstm_models,
                'xgboost': product_id in self.xgboost_models
            },
            'model_performance': {},
            'dependencies': {
                'prophet': PROPHET_AVAILABLE,
                'tensorflow': TF_AVAILABLE,
                'xgboost': XGBOOST_AVAILABLE
            }
        }
        
        # Add performance metrics
        for model_type in ['arima', 'prophet', 'lstm', 'xgboost']:
            key = f'{product_id}_{model_type}'
            if key in self.model_performance:
                info['model_performance'][model_type] = self.model_performance[key]
        
        return info