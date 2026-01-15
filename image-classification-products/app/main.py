from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import base64
import io
from PIL import Image
import numpy as np
import json
from datetime import datetime
import time

from .model import ImageClassificationModel
from .schemas import (
    ClassificationRequest, ClassificationResponse, BatchClassificationRequest,
    BatchClassificationResponse, ModelInfo, TrainingRequest, TrainingResponse,
    EvaluationRequest, EvaluationResponse, ProductCategory, ModelType
)

app = FastAPI(
    title="Image Classification for Products API",
    description="API for classifying product images using CNN and EfficientNet models",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

def get_model():
    """Get or initialize the classification model"""
    global model
    if model is None:
        model = ImageClassificationModel()
        model.load_model()
    return model

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Image Classification for Products API",
        "version": "1.0.0",
        "status": "active",
        "available_models": [model_type.value for model_type in ModelType],
        "available_categories": [category.value for category in ProductCategory]
    }

@app.post("/classify", response_model=ClassificationResponse)
async def classify_image(request: ClassificationRequest):
    """
    Classify a single product image
    
    Args:
        request: Classification request with image data and parameters
        
    Returns:
        Classification results with predictions
    """
    try:
        start_time = time.time()
        
        # Decode base64 image
        image_data = base64.b64decode(request.image_data)
        image = Image.open(io.BytesIO(image_data))
        
        # Get model and classify
        model = get_model()
        predictions = model.classify_image(
            image,
            model_type=request.model_type,
            top_k=request.top_k,
            confidence_threshold=request.confidence_threshold,
            image_size=request.image_size
        )
        
        processing_time = time.time() - start_time
        
        return ClassificationResponse(
            predictions=predictions,
            model_type=request.model_type.value,
            processing_time=processing_time,
            image_size=request.image_size.value,
            confidence_threshold=request.confidence_threshold,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

@app.post("/classify/batch", response_model=BatchClassificationResponse)
async def classify_batch(request: BatchClassificationRequest):
    """
    Classify multiple product images in batch
    
    Args:
        request: Batch classification request
        
    Returns:
        Batch classification results
    """
    try:
        start_time = time.time()
        results = []
        
        model = get_model()
        
        for i, image_data in enumerate(request.images):
            try:
                # Decode image
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                
                # Classify
                predictions = model.classify_image(
                    image,
                    model_type=request.model_type,
                    top_k=request.top_k,
                    confidence_threshold=request.confidence_threshold,
                    image_size=request.image_size
                )
                
                results.append({
                    "index": i,
                    "predictions": predictions,
                    "status": "success"
                })
                
            except Exception as e:
                results.append({
                    "index": i,
                    "predictions": [],
                    "status": "error",
                    "error": str(e)
                })
        
        total_processing_time = time.time() - start_time
        
        return BatchClassificationResponse(
            results=results,
            total_images=len(request.images),
            successful_classifications=len([r for r in results if r["status"] == "success"]),
            model_type=request.model_type.value,
            total_processing_time=total_processing_time,
            average_processing_time=total_processing_time / len(request.images),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch classification error: {str(e)}")

@app.post("/classify/upload")
async def classify_upload(
    file: UploadFile = File(...),
    model_type: ModelType = ModelType.EFFICIENTNET_B0,
    top_k: int = 5,
    confidence_threshold: float = 0.1,
    image_size: str = "small"
):
    """
    Classify an uploaded product image file
    
    Args:
        file: Uploaded image file
        model_type: Model architecture to use
        top_k: Number of top predictions
        confidence_threshold: Minimum confidence threshold
        image_size: Image size for processing
        
    Returns:
        Classification results
    """
    try:
        start_time = time.time()
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Get model and classify
        model = get_model()
        
        # Map string to ImageSize enum
        from .schemas import ImageSize
        size_enum = ImageSize(image_size.upper())
        
        predictions = model.classify_image(
            image,
            model_type=model_type,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
            image_size=size_enum
        )
        
        processing_time = time.time() - start_time
        
        return ClassificationResponse(
            predictions=predictions,
            model_type=model_type.value,
            processing_time=processing_time,
            image_size=image_size,
            confidence_threshold=confidence_threshold,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload classification error: {str(e)}")

@app.get("/models", response_model=List[ModelInfo])
async def get_available_models():
    """Get information about available models"""
    try:
        model = get_model()
        return model.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    """
    Train or fine-tune a classification model
    
    Args:
        request: Training request with parameters
        
    Returns:
        Training results
    """
    try:
        start_time = time.time()
        
        model = get_model()
        
        # Decode training images
        training_data = []
        for item in request.training_data:
            image_data = base64.b64decode(item.image_data)
            image = Image.open(io.BytesIO(image_data))
            training_data.append({
                "image": image,
                "label": item.label,
                "category": item.category
            })
        
        # Train model
        training_results = model.train_model(
            training_data=training_data,
            model_type=request.model_type,
            epochs=request.epochs,
            learning_rate=request.learning_rate,
            validation_split=request.validation_split
        )
        
        training_time = time.time() - start_time
        
        return TrainingResponse(
            model_type=request.model_type.value,
            epochs_completed=training_results["epochs_completed"],
            final_accuracy=training_results["final_accuracy"],
            final_loss=training_results["final_loss"],
            training_samples=len(training_data),
            training_time=training_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_model(request: EvaluationRequest):
    """
    Evaluate model performance on test data
    
    Args:
        request: Evaluation request with test data
        
    Returns:
        Evaluation metrics
    """
    try:
        start_time = time.time()
        
        model = get_model()
        
        # Decode test images
        test_data = []
        for item in request.test_data:
            image_data = base64.b64decode(item.image_data)
            image = Image.open(io.BytesIO(image_data))
            test_data.append({
                "image": image,
                "true_label": item.true_label,
                "true_category": item.true_category
            })
        
        # Evaluate model
        evaluation_results = model.evaluate_model(
            test_data=test_data,
            model_type=request.model_type
        )
        
        evaluation_time = time.time() - start_time
        
        return EvaluationResponse(
            model_type=request.model_type.value,
            accuracy=evaluation_results["accuracy"],
            precision=evaluation_results["precision"],
            recall=evaluation_results["recall"],
            f1_score=evaluation_results["f1_score"],
            confusion_matrix=evaluation_results["confusion_matrix"],
            test_samples=len(test_data),
            evaluation_time=evaluation_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        model = get_model()
        return {
            "status": "healthy",
            "model_loaded": model.model is not None,
            "available_categories": [category.value for category in ProductCategory],
            "available_models": [model_type.value for model_type in ModelType],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)