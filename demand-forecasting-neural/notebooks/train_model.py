"""
Interactive Training Script for Demand Forecasting with Neural Networks
This script provides a command-line interface for training and comparing different neural network models
for demand forecasting.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import argparse
from app.model import DemandForecastingNeuralModel

def generate_synthetic_demand_data(n_days: int = 1000, product_id: str = "PROD_001") -> pd.DataFrame:
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
        'product_id': product_id
    })
    
    return df

def plot_training_history(training_history: dict, model_name: str):
    """Plot training and validation loss history"""
    
    plt.figure(figsize=(12, 5))
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(training_history['train_losses'], label='Training Loss')
    plt.plot(training_history['test_losses'], label='Validation Loss')
    plt.title(f'{model_name} - Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_forecasts(actual: list, forecasts: dict, title: str = "Demand Forecasting Comparison"):
    """Plot actual vs predicted forecasts"""
    
    plt.figure(figsize=(15, 8))
    
    # Plot actual values
    plt.subplot(2, 1, 1)
    plt.plot(actual, label='Actual Demand', color='black', linewidth=2)
    
    # Plot forecasts from different models
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (model_name, forecast) in enumerate(forecasts.items()):
        plt.plot(forecast, label=f'{model_name} Forecast', color=colors[i % len(colors)], alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Demand')
    plt.legend()
    plt.grid(True)
    
    # Plot residuals
    plt.subplot(2, 1, 2)
    for i, (model_name, forecast) in enumerate(forecasts.items()):
        residuals = np.array(actual) - np.array(forecast)
        plt.plot(residuals, label=f'{model_name} Residuals', color=colors[i % len(colors)], alpha=0.7)
    
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('Forecast Residuals')
    plt.xlabel('Time Steps')
    plt.ylabel('Residuals')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def compare_models_performance(performance_results: dict):
    """Compare performance metrics across different models"""
    
    models = list(performance_results.keys())
    metrics = ['rmse', 'mae', 'mse']
    
    # Create comparison dataframe
    comparison_data = []
    for model in models:
        model_perf = performance_results[model]
        comparison_data.append([
            model_perf.get('rmse', 0),
            model_perf.get('mae', 0),
            model_perf.get('mse', 0)
        ])
    
    df_comparison = pd.DataFrame(comparison_data, index=models, columns=['RMSE', 'MAE', 'MSE'])
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(['RMSE', 'MAE', 'MSE']):
        df_comparison[metric].plot(kind='bar', ax=axes[i], color='skyblue')
        axes[i].set_title(f'{metric} Comparison')
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return df_comparison

def train_single_model(model, df: pd.DataFrame, model_type: str, **kwargs):
    """Train a single model and return performance results"""
    
    print(f"\n{'='*50}")
    print(f"Training {model_type.upper()} Model")
    print(f"{'='*50}")
    
    start_time = datetime.now()
    
    # Train the model based on type
    if model_type == 'lstm':
        performance = model.train_lstm_model(df, **kwargs)
    elif model_type == 'gru':
        performance = model.train_gru_model(df, **kwargs)
    elif model_type == 'transformer':
        performance = model.train_transformer_model(df, **kwargs)
    elif model_type == 'cnn_lstm':
        performance = model.train_cnn_lstm_model(df, **kwargs)
    elif model_type == 'bilstm':
        performance = model.train_bilstm_model(df, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    print(f"\n{model_type.upper()} Model Training Completed!")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"RMSE: {performance['rmse']:.2f}")
    print(f"MAE: {performance['mae']:.2f}")
    print(f"MSE: {performance['mse']:.2f}")
    
    return performance, training_time

def main():
    """Main function for interactive training"""
    
    parser = argparse.ArgumentParser(description='Interactive Training for Demand Forecasting')
    parser.add_argument('--n_days', type=int, default=1000, help='Number of days of synthetic data to generate')
    parser.add_argument('--models', nargs='+', default=['lstm', 'gru'], 
                       choices=['lstm', 'gru', 'transformer', 'cnn_lstm', 'bilstm'],
                       help='Models to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--plot', action='store_true', help='Plot training results')
    parser.add_argument('--save_models', action='store_true', help='Save trained models')
    parser.add_argument('--compare', action='store_true', help='Compare all trained models')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Demand Forecasting with Neural Networks - Interactive Training")
    print("="*60)
    
    # Initialize the forecasting model
    forecasting_model = DemandForecastingNeuralModel()
    
    # Generate synthetic data
    print(f"\nGenerating {args.n_days} days of synthetic demand data...")
    df = generate_synthetic_demand_data(args.n_days)
    print(f"Data generated: {len(df)} records")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Demand statistics:")
    print(f"  Mean: {df['demand'].mean():.2f}")
    print(f"  Std: {df['demand'].std():.2f}")
    print(f"  Min: {df['demand'].min():.2f}")
    print(f"  Max: {df['demand'].max():.2f}")
    
    # Training parameters
    training_kwargs = {
        'epochs': args.epochs,
        'learning_rate': 0.001
    }
    
    # Train models
    performance_results = {}
    training_times = {}
    
    for model_type in args.models:
        try:
            performance, training_time = train_single_model(
                forecasting_model, df, model_type, **training_kwargs
            )
            performance_results[model_type] = performance
            training_times[model_type] = training_time
            
            # Plot training history if requested
            if args.plot:
                plot_training_history(performance['training_history'], model_type.upper())
                
        except Exception as e:
            print(f"Error training {model_type}: {e}")
            continue
    
    # Compare models if requested
    if args.compare and len(performance_results) > 1:
        print("\n" + "="*60)
        print("Model Performance Comparison")
        print("="*60)
        
        comparison_df = compare_models_performance(performance_results)
        print("\nPerformance Summary:")
        print(comparison_df)
        
        # Plot comparison
        if args.plot:
            compare_models_performance(performance_results)
    
    # Test forecasting with trained models
    if len(performance_results) > 0:
        print("\n" + "="*60)
        print("Testing Forecasting with Trained Models")
        print("="*60)
        
        # Use last 50 days for testing
        test_data = df.tail(50)
        historical_data = test_data['demand'].tolist()
        
        forecasts = {}
        
        for model_type in performance_results.keys():
            try:
                forecast = forecasting_model.forecast_demand(
                    model_type, historical_data, forecast_horizon=14
                )
                forecasts[model_type] = forecast
                print(f"{model_type.upper()}: Generated {len(forecast)} day forecast")
            except Exception as e:
                print(f"Error forecasting with {model_type}: {e}")
        
        # Plot forecasts if requested
        if args.plot and len(forecasts) > 0:
            plot_forecasts(historical_data[-30:], forecasts, "Demand Forecasting Comparison")
    
    # Save models if requested
    if args.save_models and len(performance_results) > 0:
        print("\n" + "="*60)
        print("Saving Trained Models")
        print("="*60)
        
        os.makedirs("saved_models", exist_ok=True)
        
        for model_type in performance_results.keys():
            try:
                filepath = f"saved_models/{model_type}_model.pth"
                forecasting_model.save_model(filepath, model_type)
                print(f"Saved {model_type} model to {filepath}")
            except Exception as e:
                print(f"Error saving {model_type} model: {e}")
    
    print("\n" + "="*60)
    print("Interactive Training Completed!")
    print("="*60)

if __name__ == "__main__":
    main()