import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SalesForecastingModel:
    def __init__(self):
        self.prophet_model = None
        self.arima_model = None
        self.linear_model = None
        self.scaler = None
        self.feature_names = []
        self.training_data = None
        
    def generate_synthetic_sales_data(self, n_days=730, start_date='2022-01-01'):
        """Generate synthetic sales data with trends, seasonality, and noise"""
        np.random.seed(42)
        
        # Create date range
        dates = pd.date_range(start=start_date, periods=n_days, freq='D')
        
        # Generate sales data with trend, seasonality, and noise
        trend = np.linspace(1000, 2000, n_days)  # Upward trend
        
        # Weekly seasonality (higher sales on weekends)
        weekly_seasonality = 200 * np.sin(2 * np.pi * np.arange(n_days) / 7)
        
        # Monthly seasonality
        monthly_seasonality = 100 * np.sin(2 * np.pi * np.arange(n_days) / 30)
        
        # Yearly seasonality (holiday peaks)
        yearly_seasonality = 300 * np.sin(2 * np.pi * np.arange(n_days) / 365)
        
        # Random noise
        noise = np.random.normal(0, 50, n_days)
        
        # Combine components
        sales = trend + weekly_seasonality + monthly_seasonality + yearly_seasonality + noise
        sales = np.maximum(sales, 100)  # Ensure positive sales
        
        # Add some promotional spikes
        promo_days = np.random.choice(n_days, size=20, replace=False)
        sales[promo_days] *= 1.5
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'sales': sales
        })
        
        return df
    
    def prepare_data(self, df):
        """Prepare data for modeling"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df.reset_index(drop=True)
        
        # Create additional features for linear regression
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # Create lag features
        df['sales_lag_1'] = df['sales'].shift(1)
        df['sales_lag_7'] = df['sales'].shift(7)
        df['sales_lag_30'] = df['sales'].shift(30)
        
        # Create rolling averages
        df['sales_ma_7'] = df['sales'].rolling(window=7).mean()
        df['sales_ma_30'] = df['sales'].rolling(window=30).mean()
        
        # Drop rows with NaN values
        df = df.dropna()
        
        return df
    
    def train_prophet_model(self, df):
        """Train Prophet model"""
        print("Training Prophet model...")
        
        # Prepare data for Prophet (must have 'ds' and 'y' columns)
        prophet_df = df[['date', 'sales']].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Initialize and train Prophet
        self.prophet_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='additive'
        )
        
        self.prophet_model.fit(prophet_df)
        
        # Store training data for future forecasts
        self.training_data = prophet_df
        
        print("Prophet model training completed!")
        
    def train_arima_model(self, df):
        """Train ARIMA model"""
        print("Training ARIMA model...")
        
        # Prepare time series data
        ts_data = df.set_index('date')['sales']
        
        # Fit ARIMA model (order can be optimized based on data)
        try:
            self.arima_model = ARIMA(ts_data, order=(1, 1, 1))
            self.arima_model = self.arima_model.fit()
            print("ARIMA model training completed!")
        except Exception as e:
            print(f"ARIMA training failed: {e}")
            self.arima_model = None
    
    def train_linear_regression_model(self, df):
        """Train Linear Regression model with polynomial features"""
        print("Training Linear Regression model...")
        
        # Prepare features
        feature_cols = [
            'day_of_week', 'month', 'year', 'day_of_year',
            'sales_lag_1', 'sales_lag_7', 'sales_lag_30',
            'sales_ma_7', 'sales_ma_30'
        ]
        
        # Create polynomial features
        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        X = poly_features.fit_transform(df[feature_cols])
        y = df['sales'].values
        
        # Train model
        self.linear_model = LinearRegression()
        self.linear_model.fit(X, y)
        
        # Store feature information
        self.scaler = poly_features
        self.feature_names = feature_cols
        
        print("Linear Regression model training completed!")
    
    def forecast_prophet(self, periods=30):
        """Generate forecast using Prophet"""
        if self.prophet_model is None:
            raise ValueError("Prophet model not trained")
        
        # Create future dataframe
        future = self.prophet_model.make_future_dataframe(periods=periods)
        
        # Make predictions
        forecast = self.prophet_model.predict(future)
        
        # Extract relevant columns
        forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        
        # Convert to expected format
        result = []
        for _, row in forecast_df.iterrows():
            result.append({
                'date': row['ds'].strftime('%Y-%m-%d'),
                'sales': max(row['yhat'], 0)  # Ensure non-negative sales
            })
        
        # Get confidence intervals
        confidence_intervals = {
            'lower': forecast_df['yhat_lower'].tolist(),
            'upper': forecast_df['yhat_upper'].tolist()
        }
        
        return result, confidence_intervals
    
    def forecast_arima(self, periods=30):
        """Generate forecast using ARIMA"""
        if self.arima_model is None:
            raise ValueError("ARIMA model not trained")
        
        # Generate forecast
        forecast = self.arima_model.forecast(steps=periods)
        
        # Create date range for forecast
        last_date = self.training_data['ds'].max()
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods)
        
        # Convert to expected format
        result = []
        for date, sales in zip(forecast_dates, forecast):
            result.append({
                'date': date.strftime('%Y-%m-%d'),
                'sales': max(sales, 0)  # Ensure non-negative sales
            })
        
        return result, None  # ARIMA doesn't provide confidence intervals by default
    
    def forecast_linear_regression(self, df, periods=30):
        """Generate forecast using Linear Regression"""
        if self.linear_model is None:
            raise ValueError("Linear Regression model not trained")
        
        # Create future data
        last_date = df['date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods)
        
        # Create features for future dates
        future_df = pd.DataFrame({'date': future_dates})
        future_df['day_of_week'] = future_df['date'].dt.dayofweek
        future_df['month'] = future_df['date'].dt.month
        future_df['year'] = future_df['date'].dt.year
        future_df['day_of_year'] = future_df['date'].dt.dayofyear
        
        # Use last known values for lag features (simplified approach)
        last_row = df.iloc[-1]
        future_df['sales_lag_1'] = last_row['sales']
        future_df['sales_lag_7'] = last_row['sales']
        future_df['sales_lag_30'] = last_row['sales']
        future_df['sales_ma_7'] = last_row['sales']
        future_df['sales_ma_30'] = last_row['sales']
        
        # Prepare features
        feature_cols = [
            'day_of_week', 'month', 'year', 'day_of_year',
            'sales_lag_1', 'sales_lag_7', 'sales_lag_30',
            'sales_ma_7', 'sales_ma_30'
        ]
        
        X_future = self.scaler.transform(future_df[feature_cols])
        
        # Make predictions
        predictions = self.linear_model.predict(X_future)
        
        # Convert to expected format
        result = []
        for date, sales in zip(future_dates, predictions):
            result.append({
                'date': date.strftime('%Y-%m-%d'),
                'sales': max(sales, 0)  # Ensure non-negative sales
            })
        
        return result, None
    
    def evaluate_model(self, model_type, df, train_size=0.8):
        """Evaluate model performance"""
        split_idx = int(len(df) * train_size)
        train_df = df[:split_idx]
        test_df = df[split_idx:]
        
        if model_type == 'prophet':
            # Train on training data
            self.train_prophet_model(train_df)
            
            # Forecast on test period
            forecast_period = len(test_df)
            forecast, _ = self.forecast_prophet(forecast_period)
            
            # Calculate metrics
            actual_sales = test_df['sales'].values
            predicted_sales = [f['sales'] for f in forecast]
            
        elif model_type == 'arima':
            # Train on training data
            self.train_arima_model(train_df)
            
            # Forecast on test period
            forecast_period = len(test_df)
            forecast, _ = self.forecast_arima(forecast_period)
            
            # Calculate metrics
            actual_sales = test_df['sales'].values
            predicted_sales = [f['sales'] for f in forecast]
            
        elif model_type == 'linear_regression':
            # Prepare data
            train_prepared = self.prepare_data(train_df)
            test_prepared = self.prepare_data(test_df)
            
            # Train on training data
            self.train_linear_regression_model(train_prepared)
            
            # Forecast on test period
            forecast_period = len(test_df)
            forecast, _ = self.forecast_linear_regression(train_prepared, forecast_period)
            
            # Calculate metrics
            actual_sales = test_prepared['sales'].values
            predicted_sales = [f['sales'] for f in forecast]
        
        # Calculate evaluation metrics
        rmse = np.sqrt(mean_squared_error(actual_sales, predicted_sales))
        mae = mean_absolute_error(actual_sales, predicted_sales)
        mape = np.mean(np.abs((actual_sales - predicted_sales) / actual_sales)) * 100
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'actual_sales': actual_sales.tolist(),
            'predicted_sales': predicted_sales
        }
    
    def save_model(self, filepath, model_type='prophet'):
        """Save the trained model"""
        if model_type == 'prophet':
            model_data = {
                'model': self.prophet_model,
                'training_data': self.training_data,
                'model_type': 'prophet'
            }
        elif model_type == 'arima':
            model_data = {
                'model': self.arima_model,
                'model_type': 'arima'
            }
        elif model_type == 'linear_regression':
            model_data = {
                'model': self.linear_model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_type': 'linear_regression'
            }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        model_type = model_data['model_type']
        
        if model_type == 'prophet':
            self.prophet_model = model_data['model']
            self.training_data = model_data['training_data']
        elif model_type == 'arima':
            self.arima_model = model_data['model']
        elif model_type == 'linear_regression':
            self.linear_model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
        
        print(f"Model loaded from {filepath}")

# Training script
if __name__ == "__main__":
    # Initialize model
    sales_model = SalesForecastingModel()
    
    # Generate training data
    print("Generating synthetic sales data...")
    df = sales_model.generate_synthetic_sales_data(n_days=730)
    
    # Prepare data
    df_prepared = sales_model.prepare_data(df)
    
    # Train Prophet model
    print("\\nTraining Prophet model...")
    sales_model.train_prophet_model(df_prepared)
    sales_model.save_model('sales_forecast_prophet.pkl', 'prophet')
    
    # Train ARIMA model
    print("\\nTraining ARIMA model...")
    sales_model.train_arima_model(df_prepared)
    sales_model.save_model('sales_forecast_arima.pkl', 'arima')
    
    # Train Linear Regression model
    print("\\nTraining Linear Regression model...")
    sales_model.train_linear_regression_model(df_prepared)
    sales_model.save_model('sales_forecast_linear.pkl', 'linear_regression')
    
    print("\\nAll models trained and saved successfully!")