from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
import uvicorn
from datetime import datetime, timedelta

from app.schemas import (
    StockData, StockPredictionRequest, StockPrediction, 
    BatchStockPrediction, ModelInfo, TrainingRequest, TrainingResponse
)
from app.model import StockMovementPredictor

# Initialize FastAPI app
app = FastAPI(
    title="Stock Price Movement Classifier API",
    description="API for predicting stock price movements using LSTM and traditional ML models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the predictor
predictor = StockMovementPredictor()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Stock Price Movement Classifier API",
        "version": "1.0.0",
        "available_models": ["lstm", "random_forest", "logistic_regression"],
        "endpoints": [
            "/predict",
            "/train",
            "/models/info",
            "/health"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "lstm_models": len(predictor.lstm_models),
            "traditional_models": len(predictor.traditional_models)
        }
    }

@app.post("/predict", response_model=StockPrediction)
async def predict_stock_movement(request: StockPredictionRequest):
    """Predict stock price movement for a single stock"""
    try:
        symbol = request.symbol.upper()
        
        if request.model_type == "lstm":
            if symbol not in predictor.lstm_models:
                raise HTTPException(
                    status_code=404, 
                    detail=f"LSTM model for {symbol} not found. Please train the model first using /train endpoint."
                )
            prediction = predictor.predict_with_lstm(symbol, request.days_ahead)
            
        elif request.model_type in ["random_forest", "logistic_regression"]:
            model_key = f"{symbol}_{request.model_type}"
            if model_key not in predictor.traditional_models:
                raise HTTPException(
                    status_code=404, 
                    detail=f"{request.model_type} model for {symbol} not found. Please train the model first using /train endpoint."
                )
            prediction = predictor.predict_with_traditional(symbol, request.model_type)
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model type: {request.model_type}")
        
        # Add additional metadata
        prediction['prediction_date'] = datetime.now().strftime('%Y-%m-%d')
        target_date = datetime.now() + timedelta(days=request.days_ahead)
        prediction['target_date'] = target_date.strftime('%Y-%m-%d')
        
        return StockPrediction(**prediction)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchStockPrediction)
async def predict_batch_stocks(
    symbols: List[str],
    model_type: str = Query(default="lstm", description="Model type to use"),
    days_ahead: int = Query(default=1, ge=1, le=30, description="Days ahead to predict")
):
    """Predict stock price movements for multiple stocks"""
    try:
        predictions = []
        up_count = 0
        down_count = 0
        total_confidence = 0
        
        for symbol in symbols:
            symbol = symbol.upper()
            request = StockPredictionRequest(
                symbol=symbol,
                days_ahead=days_ahead,
                model_type=model_type
            )
            
            try:
                prediction = await predict_stock_movement(request)
                predictions.append(prediction)
                
                if prediction.predicted_direction == "Up":
                    up_count += 1
                else:
                    down_count += 1
                    
                total_confidence += prediction.confidence
                
            except HTTPException:
                # Skip stocks that don't have trained models
                continue
        
        if not predictions:
            raise HTTPException(status_code=404, detail="No valid predictions could be made")
        
        average_confidence = total_confidence / len(predictions) if predictions else 0
        
        summary = {
            "total_predictions": len(predictions),
            "up_predictions": up_count,
            "down_predictions": down_count
        }
        
        return BatchStockPrediction(
            predictions=predictions,
            summary=summary,
            average_confidence=average_confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Batch prediction error: {str(e)}')

@app.post("/train", response_model=TrainingResponse)
async def train_models(request: TrainingRequest):
    """Train models for specified stocks"""
    try:
        start_time = datetime.now()
        models_trained = []
        best_accuracy = 0
        
        for symbol in request.symbols:
            symbol = symbol.upper()
            
            try:
                if request.model_type == "lstm":
                    # Train LSTM model
                    metrics = predictor.train_lstm_model(symbol, epochs=request.epochs)
                    model_name = f"{symbol}_lstm"
                    
                elif request.model_type in ["random_forest", "logistic_regression"]:
                    # Train traditional model
                    metrics = predictor.train_traditional_model(symbol, request.model_type)
                    model_name = f"{symbol}_{request.model_type}"
                    
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown model type: {request.model_type}")
                
                models_trained.append(model_name)
                
                if metrics['accuracy'] > best_accuracy:
                    best_accuracy = metrics['accuracy']
                
            except Exception as e:
                # Log error but continue with other symbols
                print(f"Error training {symbol}: {str(e)}")
                continue
        
        if not models_trained:
            raise HTTPException(status_code=500, detail="No models could be trained successfully")
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        return TrainingResponse(
            message=f"Successfully trained {len(models_trained)} models",
            models_trained=models_trained,
            training_time=training_time,
            best_accuracy=best_accuracy
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@app.get("/models/info", response_model=List[ModelInfo])
async def get_models_info():
    """Get information about trained models"""
    try:
        models_info = []
        
        # LSTM models info
        for symbol, model in predictor.lstm_models.items():
            model_info = ModelInfo(
                model_type="lstm",
                training_date=datetime.now().strftime('%Y-%m-%d'),
                features_used=predictor.feature_columns,
                model_parameters={
                    "hidden_units": [64, 32, 16],
                    "dropout": 0.2,
                    "sequence_length": predictor.sequence_length
                }
            )
            models_info.append(model_info)
        
        # Traditional models info
        for model_key, model in predictor.traditional_models.items():
            symbol, model_type = model_key.split('_', 1)
            model_info = ModelInfo(
                model_type=model_type,
                training_date=datetime.now().strftime('%Y-%m-%d'),
                features_used=predictor.feature_columns,
                model_parameters={
                    "n_estimators": 100 if model_type == "random_forest" else None,
                    "max_depth": None,
                    "random_state": 42
                }
            )
            models_info.append(model_info)
        
        return models_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting models info: {str(e)}")

@app.get("/stock/{symbol}/data")
async def get_stock_data(symbol: str, period: str = Query(default="1y", description="Time period for data")):
    """Get historical stock data"""
    try:
        symbol = symbol.upper()
        df = predictor.fetch_stock_data(symbol, period)
        
        # Convert to list of dictionaries
        data = []
        for _, row in df.iterrows():
            data.append({
                "symbol": symbol,
                "date": row['Date'],
                "open_price": float(row['Open']),
                "high_price": float(row['High']),
                "low_price": float(row['Low']),
                "close_price": float(row['Close']),
                "volume": int(row['Volume'])
            })
        
        return {
            "symbol": symbol,
            "period": period,
            "data_points": len(data),
            "data": data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stock data: {str(e)}")

@app.post("/models/save")
async def save_models(filepath: str = Query(default="stock_models.joblib", description="Filepath to save models")):
    """Save trained models to file"""
    try:
        predictor.save_models(filepath)
        return {"message": f"Models saved successfully to {filepath}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving models: {str(e)}")

@app.post("/models/load")
async def load_models(filepath: str = Query(default="stock_models.joblib", description="Filepath to load models from")):
    """Load trained models from file"""
    try:
        predictor.load_models(filepath)
        return {"message": f"Models loaded successfully from {filepath}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)