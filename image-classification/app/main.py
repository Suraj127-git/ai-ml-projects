from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import uvicorn
import base64
import json
import os
from datetime import datetime

from app.schemas import (
    ImageClassificationRequest, ImageClassificationResponse, ClassificationPrediction,
    ModelTrainingRequest, ModelTrainingResponse, ModelInfo, BatchClassificationRequest,
    BatchClassificationResponse, HealthResponse, ClassificationModel, DatasetName
)
from app.model import ImageClassificationModel

# Initialize FastAPI app
app = FastAPI(
    title="Image Classification API",
    description="CNN-based Image Classification API with support for multiple architectures",
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

# Initialize model
image_model = ImageClassificationModel()

# Model storage directory
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    """Load existing model if available"""
    try:
        model_path = os.path.join(MODEL_DIR, "image_classifier")
        if os.path.exists(f"{model_path}_model.h5"):
            image_model.load_model(model_path)
            print(f"Loaded existing model: {image_model.model_name}")
    except Exception as e:
        print(f"Could not load existing model: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Image Classification API",
        "version": "1.0.0",
        "status": "running",
        "models_available": ["cnn_custom", "resnet18", "resnet50", "mobilenet"],
        "datasets_available": ["cifar10", "cifar100", "custom"]
    }

@app.post("/predict", response_model=ImageClassificationResponse)
async def classify_image(request: ImageClassificationRequest):
    """Classify a single image"""
    try:
        if not image_model.is_trained:
            raise HTTPException(status_code=400, detail="No model is trained. Please train a model first using /train endpoint.")
        
        # Make prediction
        result = image_model.predict(
            request.image_base64,
            confidence_threshold=request.confidence_threshold,
            top_k=request.top_k
        )
        
        # Convert predictions to response format
        predictions = [
            ClassificationPrediction(**pred) for pred in result['predictions']
        ]
        
        return ImageClassificationResponse(
            predictions=predictions,
            model_name=result['model_name'],
            image_width=result['image_width'],
            image_height=result['image_height'],
            processing_time=result['processing_time'],
            timestamp=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

@app.post("/predict/batch", response_model=BatchClassificationResponse)
async def classify_batch(request: BatchClassificationRequest):
    """Classify multiple images in batch"""
    try:
        if not image_model.is_trained:
            raise HTTPException(status_code=400, detail="No model is trained. Please train a model first using /train endpoint.")
        
        start_time = datetime.now()
        batch_predictions = []
        
        for image_base64 in request.images_base64:
            try:
                result = image_model.predict(
                    image_base64,
                    confidence_threshold=request.confidence_threshold,
                    top_k=request.top_k
                )
                
                predictions = [
                    ClassificationPrediction(**pred) for pred in result['predictions']
                ]
                
                response = ImageClassificationResponse(
                    predictions=predictions,
                    model_name=result['model_name'],
                    image_width=result['image_width'],
                    image_height=result['image_height'],
                    processing_time=result['processing_time'],
                    timestamp=datetime.now()
                )
                
                batch_predictions.append(response)
                
            except Exception as e:
                # Handle individual image errors gracefully
                error_response = ImageClassificationResponse(
                    predictions=[],
                    model_name=request.model_name.value,
                    image_width=0,
                    image_height=0,
                    processing_time=0.0,
                    timestamp=datetime.now()
                )
                batch_predictions.append(error_response)
        
        total_time = (datetime.now() - start_time).total_seconds()
        avg_time = total_time / len(request.images_base64) if request.images_base64 else 0
        
        return BatchClassificationResponse(
            batch_predictions=batch_predictions,
            total_images=len(batch_predictions),
            total_processing_time=total_time,
            average_processing_time=avg_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch classification error: {str(e)}")

@app.post("/train", response_model=ModelTrainingResponse)
async def train_model(request: ModelTrainingRequest):
    """Train a new model"""
    try:
        # Train the model
        result = image_model.train_model(
            dataset_name=request.dataset_name.value,
            model_name=request.model_name.value,
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            validation_split=request.validation_split
        )
        
        # Save the trained model
        model_path = os.path.join(MODEL_DIR, "image_classifier")
        image_model.save_model(model_path)
        
        return ModelTrainingResponse(
            message=result['message'],
            model_name=result['model_name'],
            dataset_name=result['dataset_name'],
            epochs_completed=result['epochs_completed'],
            final_train_accuracy=result['final_train_accuracy'],
            final_val_accuracy=result.get('final_val_accuracy'),
            training_time=result['training_time'],
            metrics_history=result['metrics_history']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the current model"""
    try:
        info = image_model.get_model_info()
        
        return ModelInfo(
            model_name=info['model_name'],
            dataset_name=info['dataset_name'],
            input_shape=list(info['input_shape']),
            num_classes=info['num_classes'],
            class_names=info['class_names'],
            total_parameters=info['total_parameters'],
            model_size_mb=info['model_size_mb'],
            training_date=datetime.now() if info['is_trained'] else None,
            accuracy=None  # Could be added if we store it during training
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@app.post("/model/save")
async def save_current_model():
    """Save the current model"""
    try:
        if not image_model.is_trained:
            raise HTTPException(status_code=400, detail="No model is trained. Please train a model first.")
        
        model_path = os.path.join(MODEL_DIR, "image_classifier")
        result = image_model.save_model(model_path)
        
        return {
            "message": "Model saved successfully",
            "model_path": result['model_path'],
            "metadata_path": result['metadata_path']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving model: {str(e)}")

@app.post("/model/load")
async def load_saved_model():
    """Load a saved model"""
    try:
        model_path = os.path.join(MODEL_DIR, "image_classifier")
        if not os.path.exists(f"{model_path}_model.h5"):
            raise HTTPException(status_code=404, detail="No saved model found")
        
        metadata = image_model.load_model(model_path)
        
        return {
            "message": "Model loaded successfully",
            "model_name": metadata['model_name'],
            "dataset_name": metadata['dataset_name'],
            "training_date": metadata.get('saved_date')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=image_model.is_trained,
        available_models=["cnn_custom", "resnet18", "resnet50", "mobilenet"],
        timestamp=datetime.now()
    )

@app.get("/datasets/available")
async def get_available_datasets():
    """Get list of available datasets"""
    return {
        "datasets": [
            {
                "name": "cifar10",
                "description": "CIFAR-10 dataset with 10 classes",
                "num_classes": 10,
                "image_size": [32, 32, 3],
                "classes": ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            },
            {
                "name": "cifar100",
                "description": "CIFAR-100 dataset with 100 classes",
                "num_classes": 100,
                "image_size": [32, 32, 3],
                "classes": [f"class_{i}" for i in range(100)]
            }
        ]
    }

@app.get("/models/available")
async def get_available_models():
    """Get list of available model architectures"""
    return {
        "models": [
            {
                "name": "cnn_custom",
                "description": "Custom CNN architecture with convolutional and dense layers",
                "type": "custom"
            },
            {
                "name": "resnet18",
                "description": "ResNet-18 architecture with residual connections",
                "type": "residual"
            },
            {
                "name": "resnet50",
                "description": "ResNet-50 architecture with residual connections",
                "type": "residual"
            },
            {
                "name": "mobilenet",
                "description": "MobileNet architecture optimized for mobile devices",
                "type": "lightweight"
            }
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8007)