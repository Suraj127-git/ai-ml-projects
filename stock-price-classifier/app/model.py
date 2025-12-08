import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
import ta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class StockMovementPredictor:
    def __init__(self):
        self.lstm_models = {}
        self.traditional_models = {}
        self.scalers = {}
        self.feature_scalers = {}
        self.sequence_length = 60
        self.feature_columns = []
        
    def fetch_stock_data(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            
            if df.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Reset index to get date as column
            df.reset_index(inplace=True)
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            
            return df
        except Exception as e:
            raise Exception(f"Error fetching data for {symbol}: {str(e)}")
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to stock data"""
        # RSI
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_upper'] = bollinger.bollinger_hband()
        df['BB_lower'] = bollinger.bollinger_lband()
        df['BB_middle'] = bollinger.bollinger_mavg()
        
        # Moving Averages
        df['SMA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
        df['SMA_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
        df['EMA_12'] = ta.trend.EMAIndicator(df['Close'], window=12).ema_indicator()
        df['EMA_26'] = ta.trend.EMAIndicator(df['Close'], window=26).ema_indicator()
        
        # Volume indicators
        df['Volume_SMA'] = ta.volume.VolumeSMAIndicator(df['Close'], df['Volume'], window=20).volume_sma()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price changes and returns
        df['Price_change'] = df['Close'].pct_change()
        df['Price_change_1d'] = df['Price_change'].shift(1)
        df['Price_change_3d'] = df['Close'].pct_change(3)
        df['Price_change_5d'] = df['Close'].pct_change(5)
        
        # High-Low ratios
        df['High_Low_ratio'] = df['High'] / df['Low']
        df['Close_Open_ratio'] = df['Close'] / df['Open']
        
        # Remove rows with NaN values
        df = df.dropna()
        
        return df
    
    def create_target_variable(self, df: pd.DataFrame, days_ahead: int = 1) -> pd.DataFrame:
        """Create target variable for classification"""
        # Calculate future price change
        df[f'Future_price_{days_ahead}d'] = df['Close'].shift(-days_ahead)
        df[f'Price_change_{days_ahead}d'] = (df[f'Future_price_{days_ahead}d'] - df['Close']) / df['Close'] * 100
        
        # Create binary target: 1 if price goes up, 0 if down
        df['Target'] = (df[f'Price_change_{days_ahead}d'] > 0).astype(int)
        
        # Remove last rows where target is NaN
        df = df.dropna()
        
        return df
    
    def prepare_lstm_data(self, df: pd.DataFrame, feature_columns: List[str]) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """Prepare data for LSTM model"""
        scaler = MinMaxScaler()
        
        # Scale features
        features = df[feature_columns].values
        scaled_features = scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_features)):
            X.append(scaled_features[i-self.sequence_length:i])
            y.append(df['Target'].iloc[i])
        
        X = np.array(X)
        y = np.array(y)
        
        return X, y, scaler
    
    def build_lstm_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Build LSTM model"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, return_sequences=True),
            Dropout(0.2),
            LSTM(16),
            Dropout(0.2),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_lstm_model(self, symbol: str, epochs: int = 50, batch_size: int = 32) -> Dict:
        """Train LSTM model for a specific stock"""
        try:
            # Fetch and prepare data
            df = self.fetch_stock_data(symbol)
            df = self.add_technical_indicators(df)
            df = self.create_target_variable(df)
            
            # Select feature columns
            feature_columns = [
                'Open', 'High', 'Low', 'Volume', 'RSI', 'MACD', 'MACD_signal',
                'BB_upper', 'BB_lower', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                'Volume_ratio', 'Price_change_1d', 'Price_change_3d', 'High_Low_ratio'
            ]
            
            # Prepare LSTM data
            X, y, scaler = self.prepare_lstm_data(df, feature_columns)
            
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Build and train model
            model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
            
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=0
            )
            
            # Evaluate model
            y_pred_prob = model.predict(X_test)
            y_pred = (y_pred_prob > 0.5).astype(int)
            
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store model and scaler
            self.lstm_models[symbol] = model
            self.scalers[symbol] = scaler
            self.feature_columns = feature_columns
            
            return {
                'accuracy': accuracy,
                'model_type': 'lstm',
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'epochs_trained': epochs
            }
            
        except Exception as e:
            raise Exception(f"Error training LSTM model for {symbol}: {str(e)}")
    
    def train_traditional_model(self, symbol: str, model_type: str = 'random_forest') -> Dict:
        """Train traditional ML model"""
        try:
            # Fetch and prepare data
            df = self.fetch_stock_data(symbol)
            df = self.add_technical_indicators(df)
            df = self.create_target_variable(df)
            
            # Select features
            feature_columns = [
                'Open', 'High', 'Low', 'Volume', 'RSI', 'MACD', 'MACD_signal',
                'BB_upper', 'BB_lower', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                'Volume_ratio', 'Price_change_1d', 'Price_change_3d', 'High_Low_ratio'
            ]
            
            X = df[feature_columns]
            y = df['Target']
            
            # Split data
            split_idx = int(0.8 * len(df))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            feature_scaler = MinMaxScaler()
            X_train_scaled = feature_scaler.fit_transform(X_train)
            X_test_scaled = feature_scaler.transform(X_test)
            
            # Train model
            if model_type == 'random_forest':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_type == 'logistic_regression':
                model = LogisticRegression(random_state=42)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store model and scaler
            model_key = f"{symbol}_{model_type}"
            self.traditional_models[model_key] = model
            self.feature_scalers[model_key] = feature_scaler
            
            return {
                'accuracy': accuracy,
                'model_type': model_type,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
        except Exception as e:
            raise Exception(f"Error training {model_type} model for {symbol}: {str(e)}")
    
    def predict_with_lstm(self, symbol: str, days_ahead: int = 1) -> Dict:
        """Make prediction using LSTM model"""
        if symbol not in self.lstm_models:
            raise ValueError(f"LSTM model for {symbol} not found. Train the model first.")
        
        try:
            # Fetch recent data
            df = self.fetch_stock_data(symbol, period="6mo")
            df = self.add_technical_indicators(df)
            
            # Get the last sequence_length days
            recent_data = df.tail(self.sequence_length)
            
            # Scale features
            features = recent_data[self.feature_columns].values
            scaled_features = self.scalers[symbol].transform(features)
            
            # Reshape for LSTM
            X = np.array([scaled_features])
            
            # Make prediction
            model = self.lstm_models[symbol]
            prediction_prob = model.predict(X)[0][0]
            
            # Get current price
            current_price = df['Close'].iloc[-1]
            
            # Determine direction and confidence
            if prediction_prob > 0.5:
                direction = "Up"
                confidence = prediction_prob
            else:
                direction = "Down"
                confidence = 1 - prediction_prob
            
            return {
                'symbol': symbol,
                'predicted_direction': direction,
                'confidence': float(confidence),
                'probability_up': float(prediction_prob),
                'probability_down': float(1 - prediction_prob),
                'current_price': float(current_price),
                'model_used': 'lstm'
            }
            
        except Exception as e:
            raise Exception(f"Error making LSTM prediction for {symbol}: {str(e)}")
    
    def predict_with_traditional(self, symbol: str, model_type: str = 'random_forest') -> Dict:
        """Make prediction using traditional ML model"""
        model_key = f"{symbol}_{model_type}"
        if model_key not in self.traditional_models:
            raise ValueError(f"{model_type} model for {symbol} not found. Train the model first.")
        
        try:
            # Fetch recent data
            df = self.fetch_stock_data(symbol, period="6mo")
            df = self.add_technical_indicators(df)
            
            # Get the latest data point
            latest_data = df.tail(1)
            
            # Extract features
            feature_columns = [
                'Open', 'High', 'Low', 'Volume', 'RSI', 'MACD', 'MACD_signal',
                'BB_upper', 'BB_lower', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                'Volume_ratio', 'Price_change_1d', 'Price_change_3d', 'High_Low_ratio'
            ]
            
            X = latest_data[feature_columns].values
            
            # Scale features
            X_scaled = self.feature_scalers[model_key].transform(X)
            
            # Make prediction
            model = self.traditional_models[model_key]
            prediction_prob = model.predict_proba(X_scaled)[0][1]  # Probability of class 1 (Up)
            
            # Get current price
            current_price = df['Close'].iloc[-1]
            
            # Determine direction and confidence
            if prediction_prob > 0.5:
                direction = "Up"
                confidence = prediction_prob
            else:
                direction = "Down"
                confidence = 1 - prediction_prob
            
            return {
                'symbol': symbol,
                'predicted_direction': direction,
                'confidence': float(confidence),
                'probability_up': float(prediction_prob),
                'probability_down': float(1 - prediction_prob),
                'current_price': float(current_price),
                'model_used': model_type
            }
            
        except Exception as e:
            raise Exception(f"Error making {model_type} prediction for {symbol}: {str(e)}")
    
    def save_models(self, filepath: str):
        """Save trained models"""
        models_data = {
            'lstm_models': self.lstm_models,
            'traditional_models': self.traditional_models,
            'scal scalers': self.scalers,
            'feature_scalers': self.feature_scalers,
            'sequence_length': self.sequence_length,
            'feature_columns': self.feature_columns
        }
        joblib.dump(models_data, filepath)
    
    def load_models(self, filepath: str):
        """Load trained models"""
        models_data = joblib.load(filepath)
        self.lstm_models = models_data['lstm_models']
        self.traditional_models = models_data['traditional_models']
        self.scalers = models_data['scal scalers']
        self.feature_scalers = models_data['feature_scalers']
        self.sequence_length = models_data['sequence_length']
        self.feature_columns = models_data['feature_columns']