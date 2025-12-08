from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import logging

from .schemas import (
    FakeNewsRequest, 
    FakeNewsResponse, 
    BatchFakeNewsRequest, 
    BatchFakeNewsResponse,
    ModelInfo,
    HealthResponse,
    ModelType
)
from .model import fake_news_detector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fake News Detector API",
    description="API for detecting fake news using various NLP models",
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

@app.get("/")
async def root():
    return {
        "message": "Fake News Detector API",
        "version": "1.0.0",
        "available_models": [model.value for model in ModelType],
        "endpoints": [
            "/predict",
            "/predict/batch",
            "/models",
            "/health"
        ]
    }

@app.post("/predict", response_model=FakeNewsResponse)
async def predict_fake_news(request: FakeNewsRequest):
    try:
        # Get prediction from model
        result = fake_news_detector.predict(
            text=request.text,
            model_type=request.model_type
        )
        
        if result["prediction"] == "ERROR":
            raise HTTPException(status_code=500, detail=result.get("error", "Prediction failed"))
        
        return FakeNewsResponse(
            text=request.text,
            title=request.title,
            prediction=result["prediction"],
            confidence=result["confidence"],
            model_type=request.model_type,
            probabilities=result["probabilities"]
        )
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchFakeNewsResponse)
async def predict_batch_fake_news(request: BatchFakeNewsRequest):
    try:
        predictions = []
        
        for fake_news_request in request.texts:
            result = fake_news_detector.predict(
                text=fake_news_request.text,
                model_type=request.model_type
            )
            
            if result["prediction"] != "ERROR":
                response = FakeNewsResponse(
                    text=fake_news_request.text,
                    title=fake_news_request.title,
                    prediction=result["prediction"],
                    confidence=result["confidence"],
                    model_type=request.model_type,
                    probabilities=result["probabilities"]
                )
                predictions.append(response)
        
        return BatchFakeNewsResponse(
            predictions=predictions,
            model_type=request.model_type,
            total_processed=len(predictions)
        )
        
    except Exception as e:
        logger.error(f"Error in batch predict endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f'Batch prediction error: {str(e)}')

@app.get("/models", response_model=List[ModelInfo])
async def get_available_models():
    try:
        models_info = []
        
        for model_type in ModelType:
            model_info = fake_news_detector.get_model_info(model_type)
            models_info.append(ModelInfo(
                model_type=model_type,
                description=model_info["description"],
                accuracy=model_info["accuracy"],
                features=model_info["features"]
            ))
        
        return models_info
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model info error: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        # Check which models are loaded
        loaded_models = []
        available_models = list(ModelType)
        
        # Test loading TF-IDF model
        if fake_news_detector.load_tfidf_model():
            loaded_models.append(ModelType.TFIDF_LOGISTIC)
        
        return HealthResponse(
            status="healthy",
            models_loaded=loaded_models,
            available_models=available_models
        )
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            models_loaded=[],
            available_models=list(ModelType)
        )

@app.get("/models/{model_type}/load")
async def load_model(model_type: ModelType):
    try:
        success = False
        
        if model_type == ModelType.TFIDF_LOGISTIC:
            success = fake_news_detector.load_tfidf_model()
        elif model_type == ModelType.BERT:
            success = fake_news_detector.load_bert_model()
        elif model_type == ModelType.ROBERTA:
            success = fake_news_detector.load_roberta_model()
        
        if success:
            return {"message": f"Model {model_type} loaded successfully"}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to load model {model_type}")
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model loading error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009)