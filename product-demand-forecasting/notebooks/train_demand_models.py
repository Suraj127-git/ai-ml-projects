"""
Training and Evaluation Script for Product Demand Forecasting Models

This notebook demonstrates how to train and evaluate different forecasting models
for product demand prediction using the DemandForecastingModel class.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our forecasting model
import sys
sys.path.append('..')
from app.model import DemandForecastingModel

def main():
    print("🚀 Product Demand Forecasting Model Training")
    print("=" * 60)
    
    # Initialize model
    model = DemandForecastingModel()
    
    # Step 1: Generate Synthetic Data
    print("\n1. Generating synthetic demand data...")
    df = model.generate_synthetic_demand_data(n_products=5, n_days=730)
    print(f"✅ Generated {len(df)} records for {df['product_id'].nunique()} products")
    
    # Display sample data
    print("\nSample data:")
    print(df.head())
    
    # Step 2: Data Exploration
    print("\n2. Exploring demand patterns...")
    
    # Summary statistics by product
    summary_stats = df.groupby('product_id').agg({
        'demand': ['mean', 'std', 'min', 'max'],
        'price': 'mean',
        'promotion': 'mean',
        'holiday': 'mean'
    }).round(2)
    
    print("\nDemand Summary by Product:")
    print(summary_stats)
    
    # Step 3: Train Models for Each Product
    print("\n3. Training forecasting models...")
    
    products = df['product_id'].unique()
    training_results = {}
    
    for product_id in products[:3]:  # Train first 3 products for demo
        print(f"\n📦 Training models for {product_id}...")
        
        try:
            # Train ARIMA model
            print("   Training ARIMA...")
            arima_perf = model.train_arima_model(df, product_id)
            training_results[f"{product_id}_arima"] = arima_perf
            
            # Train Prophet model
            print("   Training Prophet...")
            prophet_perf = model.train_prophet_model(df, product_id)
            training_results[f"{product_id}_prophet"] = prophet_perf
            
            # Train XGBoost model
            print("   Training XGBoost...")
            xgboost_perf = model.train_xgboost_model(df, product_id)
            training_results[f"{product_id}_xgboost"] = xgboost_perf
            
        except Exception as e:
            print(f"   ❌ Training failed for {product_id}: {e}")
            continue
    
    # Step 4: Model Performance Comparison
    print("\n4. Model Performance Comparison:")
    print("-" * 40)
    
    performance_df = []
    for key, perf in training_results.items():
        if 'error' not in perf:
            product_id, model_type = key.rsplit('_', 1)
            performance_df.append({
                'Product': product_id,
                'Model': model_type.upper(),
                'MAPE (%)': perf.get('mape', 0),
                'RMSE': perf.get('rmse', 0),
                'MAE': perf.get('mae', 0),
                'Training Samples': perf.get('training_samples', 0)
            })
    
    if performance_df:
        perf_comparison = pd.DataFrame(performance_df)
        print(perf_comparison.to_string(index=False))
        
        # Find best model per product
        print("\n🏆 Best Model per Product (by MAPE):")
        best_models = perf_comparison.loc[perf_comparison.groupby('Product')['MAPE (%)'].idxmin()]
        for _, row in best_models.iterrows():
            print(f"   {row['Product']}: {row['Model']} (MAPE: {row['MAPE (%)']:.2f}%)")
    
    # Step 5: Generate Forecasts
    print("\n5. Generating sample forecasts...")
    
    if products:
        test_product = products[0]
        print(f"\n📈 Generating 30-day forecast for {test_product}...")
        
        # Generate forecasts with different models
        forecasts = {}
        
        try:
            # ARIMA forecast
            if f"{test_product}_arima" in training_results:
                arima_forecast = model.forecast_demand(test_product, "arima", periods=30, historical_data=df)
                forecasts['ARIMA'] = arima_forecast
                print(f"   ✅ ARIMA forecast generated")
            
            # Prophet forecast
            if f"{test_product}_prophet" in training_results:
                prophet_forecast = model.forecast_demand(test_product, "prophet", periods=30, historical_data=df)
                forecasts['Prophet'] = prophet_forecast
                print(f"   ✅ Prophet forecast generated")
            
            # XGBoost forecast
            if f"{test_product}_xgboost" in training_results:
                xgboost_forecast = model.forecast_demand(test_product, "xgboost", periods=30, historical_data=df)
                forecasts['XGBoost'] = xgboost_forecast
                print(f"   ✅ XGBoost forecast generated")
            
        except Exception as e:
            print(f"   ❌ Forecast generation failed: {e}")
    
    # Step 6: Model Information
    print("\n6. Model Information:")
    print("-" * 30)
    
    for product_id in products[:2]:
        try:
            info = model.get_model_info(product_id)
            print(f"\n📊 {product_id}:")
            print(f"   Available models: {sum(info['models_available'].values())}/4")
            print(f"   Dependencies: Prophet={info['dependencies']['prophet']}, "
                  f"TensorFlow={info['dependencies']['tensorflow']}, "
                  f"XGBoost={info['dependencies']['xgboost']}")
            
            if info['model_performance']:
                print("   Performance:")
                for model_type, perf in info['model_performance'].items():
                    print(f"     {model_type}: MAPE={perf.get('mape', 'N/A'):.2f}%")
        except Exception as e:
            print(f"   Error getting info for {product_id}: {e}")
    
    # Step 7: Save Trained Models
    print("\n7. Saving trained models...")
    try:
        if products:
            save_path = f"demand_forecast_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            model.save_model(products[0], "ensemble", save_path)
            print(f"✅ Models saved to {save_path}")
    except Exception as e:
        print(f"❌ Model saving failed: {e}")
    
    # Step 8: Demand Pattern Analysis
    print("\n8. Analyzing demand patterns...")
    
    for product_id in products[:2]:
        try:
            product_data = df[df['product_id'] == product_id].copy()
            
            # Calculate trend
            demand_values = product_data['demand'].values
            x = np.arange(len(demand_values))
            trend_slope = np.polyfit(x, demand_values, 1)[0]
            
            # Calculate seasonality
            monthly_avg = product_data.groupby(product_data['date'].dt.month)['demand'].mean()
            seasonal_variation = monthly_avg.std() / monthly_avg.mean() if len(monthly_avg) > 1 else 0
            
            # Calculate volatility
            volatility = np.std(demand_values) / np.mean(demand_values)
            
            print(f"\n📈 {product_id} Demand Pattern:")
            print(f"   Trend slope: {trend_slope:.4f}")
            print(f"   Seasonal variation: {seasonal_variation:.4f}")
            print(f"   Volatility: {volatility:.4f}")
            
        except Exception as e:
            print(f"   Error analyzing {product_id}: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Training and evaluation completed!")
    print(f"📊 Trained models for {len([k for k in training_results.keys() if 'arima' in k])} products")
    print(f"🔍 Analyzed demand patterns for {min(2, len(products))} products")
    print(f"💾 Saved models to file")
    print("\n💡 Next steps:")
    print("   - Deploy the API using app/main.py")
    print("   - Test with real demand data")
    print("   - Fine-tune model hyperparameters")
    print("   - Set up automated retraining pipeline")

if __name__ == "__main__":
    main()