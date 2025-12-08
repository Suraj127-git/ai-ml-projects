import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, List, Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class LSTMForecaster(nn.Module):
    """LSTM-based demand forecasting model"""
    
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output

class GRUForecaster(nn.Module):
    """GRU-based demand forecasting model"""
    
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        super(GRUForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        gru_out = gru_out[:, -1, :]
        gru_out = self.dropout(gru_out)
        output = self.fc(gru_out)
        return output

class TransformerForecaster(nn.Module):
    """Transformer-based demand forecasting model"""
    
    def __init__(self, input_size=1, d_model=64, nhead=4, num_encoder_layers=2, 
                 dim_feedforward=128, dropout=0.1, output_size=1):
        super(TransformerForecaster, self).__init__()
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.positional_encoding = self._generate_positional_encoding(1000, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, output_size)
        
    def _generate_positional_encoding(self, max_len, d_model):
        """Generate positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
        
    def forward(self, x):
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :].to(x.device)
        
        # Apply transformer encoder
        transformer_out = self.transformer_encoder(x)
        
        # Take the last time step
        transformer_out = transformer_out[:, -1, :]
        transformer_out = self.dropout(transformer_out)
        
        # Final prediction
        output = self.fc(transformer_out)
        return output

class CNNLSTMForecaster(nn.Module):
    """CNN-LSTM hybrid model for demand forecasting"""
    
    def __init__(self, input_size=1, cnn_filters=[32, 64], kernel_size=3, 
                 lstm_hidden_size=50, lstm_num_layers=2, output_size=1, dropout=0.2):
        super(CNNLSTMForecaster, self).__init__()
        
        # CNN layers for feature extraction
        self.cnn_layers = nn.ModuleList()
        in_channels = input_size
        for i, out_channels in enumerate(cnn_filters):
            self.cnn_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
            self.cnn_layers.append(nn.ReLU())
            self.cnn_layers.append(nn.MaxPool1d(2))
            in_channels = out_channels
        
        # LSTM layers
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm = nn.LSTM(cnn_filters[-1], lstm_hidden_size, lstm_num_layers, 
                           batch_first=True, dropout=dropout if lstm_num_layers > 1 else 0)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden_size, output_size)
        
    def forward(self, x):
        # Transpose for CNN (batch_size, channels, sequence_length)
        x = x.transpose(1, 2)
        
        # Apply CNN layers
        for layer in self.cnn_layers:
            x = layer(x)
        
        # Transpose back for LSTM (batch_size, sequence_length, features)
        x = x.transpose(1, 2)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        
        # Final prediction
        output = self.fc(lstm_out)
        return output

class BiLSTMForecaster(nn.Module):
    """Bidirectional LSTM model for demand forecasting"""
    
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        super(BiLSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, 
                               batch_first=True, dropout=dropout if num_layers > 1 else 0,
                               bidirectional=True)
        
        # Output layers - multiply by 2 for bidirectional
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        
    def forward(self, x):
        # Apply bidirectional LSTM
        bilstm_out, _ = self.bilstm(x)
        bilstm_out = bilstm_out[:, -1, :]  # Take the last time step
        bilstm_out = self.dropout(bilstm_out)
        
        # Final prediction
        output = self.fc(bilstm_out)
        return output

class DemandForecastingNeuralModel:
    """Main class for demand forecasting using neural networks"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.sequence_length = 30
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_performance = {}
        
    def create_sequences(self, data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series forecasting"""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)
        
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'demand') -> Dict[str, Any]:
        """Prepare data for neural network training"""
        
        demand_data = df[target_column].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(demand_data)
        
        X, y = self.create_sequences(scaled_data, self.sequence_length)
        
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        self.scalers['demand'] = scaler
        
        return {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler
        }
        
    def train_lstm_model(self, df: pd.DataFrame, target_column: str = 'demand',
                        hidden_size: int = 50, num_layers: int = 2, 
                        epochs: int = 100, learning_rate: float = 0.001) -> Dict[str, Any]:
        """Train LSTM model for demand forecasting"""
        
        print("Training LSTM model for demand forecasting...")
        
        data = self.prepare_data(df, target_column)
        model = LSTMForecaster(hidden_size=hidden_size, num_layers=num_layers)
        model = model.to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        train_losses = []
        test_losses = []
        
        model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in data['train_loader']:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(data['train_loader'])
            train_losses.append(avg_train_loss)
            
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in data['test_loader']:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    test_loss += loss.item()
            
            avg_test_loss = test_loss / len(data['test_loader'])
            test_losses.append(avg_test_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}')
        
        self.models['lstm'] = model
        
        # Evaluate model
        evaluation_results = self.evaluate_model(model, data['test_loader'], data['scaler'])
        
        self.model_performance['lstm'] = {
            **evaluation_results,
            'training_history': {'train_losses': train_losses, 'test_losses': test_losses},
            'model_type': 'lstm'
        }
        
        print(f"LSTM model training completed!")
        print(f"RMSE: {evaluation_results['rmse']:.2f}, MAE: {evaluation_results['mae']:.2f}")
        
        return self.model_performance['lstm']
        
    def train_gru_model(self, df: pd.DataFrame, target_column: str = 'demand',
                       hidden_size: int = 50, num_layers: int = 2, 
                       epochs: int = 100, learning_rate: float = 0.001) -> Dict[str, Any]:
        """Train GRU model for demand forecasting"""
        
        print("Training GRU model for demand forecasting...")
        
        data = self.prepare_data(df, target_column)
        model = GRUForecaster(hidden_size=hidden_size, num_layers=num_layers)
        model = model.to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        train_losses = []
        test_losses = []
        
        model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in data['train_loader']:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(data['train_loader'])
            train_losses.append(avg_train_loss)
            
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in data['test_loader']:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    test_loss += loss.item()
            
            avg_test_loss = test_loss / len(data['test_loader'])
            test_losses.append(avg_test_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}')
        
        self.models['gru'] = model
        
        # Evaluate model
        evaluation_results = self.evaluate_model(model, data['test_loader'], data['scaler'])
        
        self.model_performance['gru'] = {
            **evaluation_results,
            'training_history': {'train_losses': train_losses, 'test_losses': test_losses},
            'model_type': 'gru'
        }
        
        print(f"GRU model training completed!")
        print(f"RMSE: {evaluation_results['rmse']:.2f}, MAE: {evaluation_results['mae']:.2f}")
        
        return self.model_performance['gru']
        
    def train_transformer_model(self, df: pd.DataFrame, target_column: str = 'demand',
                               d_model: int = 64, nhead: int = 4, num_encoder_layers: int = 2,
                               dim_feedforward: int = 128, dropout: float = 0.1,
                               epochs: int = 100, learning_rate: float = 0.001) -> Dict[str, Any]:
        """Train Transformer model for demand forecasting"""
        
        print("Training Transformer model for demand forecasting...")
        
        data = self.prepare_data(df, target_column)
        model = TransformerForecaster(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                     dim_feedforward=dim_feedforward, dropout=dropout)
        model = model.to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        train_losses = []
        test_losses = []
        
        model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in data['train_loader']:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(data['train_loader'])
            train_losses.append(avg_train_loss)
            
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in data['test_loader']:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    test_loss += loss.item()
            
            avg_test_loss = test_loss / len(data['test_loader'])
            test_losses.append(avg_test_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}')
        
        self.models['transformer'] = model
        
        # Evaluate model
        evaluation_results = self.evaluate_model(model, data['test_loader'], data['scaler'])
        
        self.model_performance['transformer'] = {
            **evaluation_results,
            'training_history': {'train_losses': train_losses, 'test_losses': test_losses},
            'model_type': 'transformer'
        }
        
        print(f"Transformer model training completed!")
        print(f"RMSE: {evaluation_results['rmse']:.2f}, MAE: {evaluation_results['mae']:.2f}")
        
        return self.model_performance['transformer']
        
    def train_cnn_lstm_model(self, df: pd.DataFrame, target_column: str = 'demand',
                            cnn_filters: List[int] = [32, 64], kernel_size: int = 3,
                            lstm_hidden_size: int = 50, lstm_num_layers: int = 2,
                            epochs: int = 100, learning_rate: float = 0.001) -> Dict[str, Any]:
        """Train CNN-LSTM hybrid model for demand forecasting"""
        
        print("Training CNN-LSTM model for demand forecasting...")
        
        data = self.prepare_data(df, target_column)
        model = CNNLSTMForecaster(cnn_filters=cnn_filters, kernel_size=kernel_size,
                                 lstm_hidden_size=lstm_hidden_size, lstm_num_layers=lstm_num_layers)
        model = model.to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        train_losses = []
        test_losses = []
        
        model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in data['train_loader']:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(data['train_loader'])
            train_losses.append(avg_train_loss)
            
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in data['test_loader']:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    test_loss += loss.item()
            
            avg_test_loss = test_loss / len(data['test_loader'])
            test_losses.append(avg_test_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}')
        
        self.models['cnn_lstm'] = model
        
        # Evaluate model
        evaluation_results = self.evaluate_model(model, data['test_loader'], data['scaler'])
        
        self.model_performance['cnn_lstm'] = {
            **evaluation_results,
            'training_history': {'train_losses': train_losses, 'test_losses': test_losses},
            'model_type': 'cnn_lstm'
        }
        
        print(f"CNN-LSTM model training completed!")
        print(f"RMSE: {evaluation_results['rmse']:.2f}, MAE: {evaluation_results['mae']:.2f}")
        
        return self.model_performance['cnn_lstm']
        
    def train_bilstm_model(self, df: pd.DataFrame, target_column: str = 'demand',
                          hidden_size: int = 50, num_layers: int = 2, 
                          epochs: int = 100, learning_rate: float = 0.001) -> Dict[str, Any]:
        """Train Bidirectional LSTM model for demand forecasting"""
        
        print("Training BiLSTM model for demand forecasting...")
        
        data = self.prepare_data(df, target_column)
        model = BiLSTMForecaster(hidden_size=hidden_size, num_layers=num_layers)
        model = model.to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        train_losses = []
        test_losses = []
        
        model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in data['train_loader']:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(data['train_loader'])
            train_losses.append(avg_train_loss)
            
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in data['test_loader']:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    test_loss += loss.item()
            
            avg_test_loss = test_loss / len(data['test_loader'])
            test_losses.append(avg_test_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}')
        
        self.models['bilstm'] = model
        
        # Evaluate model
        evaluation_results = self.evaluate_model(model, data['test_loader'], data['scaler'])
        
        self.model_performance['bilstm'] = {
            **evaluation_results,
            'training_history': {'train_losses': train_losses, 'test_losses': test_losses},
            'model_type': 'bilstm'
        }
        
        print(f"BiLSTM model training completed!")
        print(f"RMSE: {evaluation_results['rmse']:.2f}, MAE: {evaluation_results['mae']:.2f}")
        
        return self.model_performance['bilstm']
        
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader, scaler: Any) -> Dict[str, float]:
        """Evaluate model performance"""
        
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                outputs = model(batch_X)
                predictions.extend(outputs.cpu().numpy().flatten())
                actuals.extend(batch_y.numpy().flatten())
        
        # Inverse transform predictions and actuals
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'predictions': predictions.tolist(),
            'actuals': actuals.tolist()
        }
        
    def forecast_demand(self, model_type: str, historical_data: List[float], 
                       forecast_horizon: int) -> List[float]:
        """Forecast demand using trained model"""
        
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained. Please train the model first.")
        
        model = self.models[model_type]
        scaler = self.scalers['demand']
        
        # Convert to numpy array and scale
        historical_array = np.array(historical_data).reshape(-1, 1)
        scaled_historical = scaler.transform(historical_array).flatten()
        
        forecasts = []
        current_sequence = scaled_historical[-self.sequence_length:].copy()
        
        model.eval()
        with torch.no_grad():
            for _ in range(forecast_horizon):
                input_seq = torch.FloatTensor(current_sequence.reshape(1, -1, 1)).to(self.device)
                prediction = model(input_seq)
                predicted_value = prediction.cpu().numpy().flatten()[0]
                
                # Inverse transform the prediction
                predicted_value = scaler.inverse_transform([[predicted_value]])[0, 0]
                forecasts.append(float(predicted_value))
                
                # Update sequence for next prediction
                current_sequence = np.append(current_sequence[1:], predicted_value)
        
        return forecasts
        
    def generate_synthetic_demand_data(self, n_days: int = 1000) -> pd.DataFrame:
        """Generate synthetic demand data for training/testing"""
        
        np.random.seed(42)
        
        # Generate dates
        start_date = pd.Timestamp('2020-01-01')
        dates = pd.date_range(start=start_date, periods=n_days, freq='D')
        
        # Generate synthetic demand with trend, seasonality, and noise
        trend = np.linspace(100, 200, n_days)
        weekly_seasonality = 20 * np.sin(2 * np.pi * np.arange(n_days) / 7)
        monthly_seasonality = 30 * np.sin(2 * np.pi * np.arange(n_days) / 30)
        yearly_seasonality = 40 * np.sin(2 * np.pi * np.arange(n_days) / 365)
        noise = np.random.normal(0, 15, n_days)
        
        demand = trend + weekly_seasonality + monthly_seasonality + yearly_seasonality + noise
        demand = np.maximum(demand, 0)
        
        # Add some promotional spikes
        promo_days = np.random.choice(n_days, size=n_days // 20, replace=False)
        demand[promo_days] *= np.random.uniform(1.5, 2.5, len(promo_days))
        
        df = pd.DataFrame({
            'date': dates,
            'demand': demand,
            'product_id': 'PROD_001'
        })
        
        return df
        
    def save_model(self, filepath: str, model_type: str):
        """Save trained model"""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained.")
        
        model_data = {
            'model_type': model_type,
            'model_state_dict': self.models[model_type].state_dict(),
            'scalers': self.scalers,
            'sequence_length': self.sequence_length,
            'model_performance': self.model_performance[model_type],
            'saved_at': pd.Timestamp.now().isoformat()
        }
        
        torch.save(model_data, filepath)
        print(f"Model {model_type} saved to {filepath}")
        
    def load_model(self, filepath: str, model_type: str):
        """Load trained model"""
        model_data = torch.load(filepath, map_location=self.device)
        
        # Initialize model architecture based on model type
        if model_type == 'lstm':
            model = LSTMForecaster()
        elif model_type == 'gru':
            model = GRUForecaster()
        elif model_type == 'transformer':
            model = TransformerForecaster()
        elif model_type == 'cnn_lstm':
            model = CNNLSTMForecaster()
        elif model_type == 'bilstm':
            model = BiLSTMForecaster()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        model.load_state_dict(model_data['model_state_dict'])
        model.to(self.device)
        
        self.models[model_type] = model
        self.scalers = model_data['scalers']
        self.sequence_length = model_data['sequence_length']
        self.model_performance[model_type] = model_data['model_performance']
        
        print(f"Model {model_type} loaded from {filepath}")
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models"""
        return {
            'available_models': list(self.models.keys()),
            'model_performance': self.model_performance,
            'sequence_length': self.sequence_length,
            'device': str(self.device)
        }