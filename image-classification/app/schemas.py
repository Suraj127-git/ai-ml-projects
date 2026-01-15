from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union
from datetime import datetime
from enum import Enum

class ClassificationModel(str, Enum):
    """Available classification models"""
    CNN_CUSTOM = "cnn_custom"
    RESNET18 = "resnet18"
    RESNET50 = "resnet50"
    VGG16 = "vgg16"
    MOBILENET = "mobilenet"
    EFFICIENTNET = "efficientnet"

class DatasetName(str, Enum):
    """Available datasets"""
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    CUSTOM = "custom"

class ImageClassificationRequest(BaseModel):
    """Request for image classification"""
    image_base64: str = Field(..., description="Base64 encoded image to classify")
    model_name: ClassificationModel = Field(default=ClassificationModel.CNN_CUSTOM, description="Model to use for classification")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")
    top_k: int = Field(default=5, ge=1, le=10, description="Number of top predictions to return")
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
                "model_name": "cnn_custom",
                "confidence_threshold": 0.5,
                "top_k": 5
            }
        }

class ClassificationPrediction(BaseModel):
    """Single classification prediction"""
    class_name: str = Field(..., description="Predicted class name")
    class_id: int = Field(..., description="Class ID/index")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    probability: float = Field(..., ge=0.0, le=1.0, description="Raw probability")

class ImageClassificationResponse(BaseModel):
    """Response for image classification"""
    predictions: List[ClassificationPrediction] = Field(..., description="Top-k predictions")
    model_name: str = Field(..., description="Model used for classification")
    image_width: int = Field(..., description="Original image width")
    image_height: int = Field(..., description="Original image height")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of prediction")

class ModelTrainingRequest(BaseModel):
    """Request for model training"""
    dataset_name: DatasetName = Field(default=DatasetName.CIFAR10, description="Dataset to train on")
    model_name: ClassificationModel = Field(default=ClassificationModel.CNN_CUSTOM, description="Model architecture to train")
    epochs: int = Field(default=10, ge=1, le=100, description="Number of training epochs")
    batch_size: int = Field(default=32, ge=8, le=256, description="Batch size for training")
    learning_rate: float = Field(default=0.001, gt=0.0, description="Learning rate")
    validation_split: float = Field(default=0.2, ge=0.0, le=0.5, description="Validation split ratio")
    
    class Config:
        json_schema_extra = {
            "example": {
                "dataset_name": "cifar10",
                "model_name": "cnn_custom",
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001,
                "validation_split": 0.2
            }
        }

class TrainingMetrics(BaseModel):
    """Training metrics"""
    epoch: int = Field(..., description="Training epoch")
    train_loss: float = Field(..., description="Training loss")
    train_accuracy: float = Field(..., description="Training accuracy")
    val_loss: Optional[float] = Field(None, description="Validation loss")
    val_accuracy: Optional[float] = Field(None, description="Validation accuracy")

class ModelTrainingResponse(BaseModel):
    """Response for model training"""
    message: str = Field(..., description="Training status message")
    model_name: str = Field(..., description="Model name")
    dataset_name: str = Field(..., description="Dataset name")
    epochs_completed: int = Field(..., description="Number of epochs completed")
    final_train_accuracy: float = Field(..., description="Final training accuracy")
    final_val_accuracy: Optional[float] = Field(None, description="Final validation accuracy")
    training_time: float = Field(..., description="Total training time in seconds")
    metrics_history: List[TrainingMetrics] = Field(..., description="Training metrics history")

class ModelInfo(BaseModel):
    """Model information"""
    model_name: str = Field(..., description="Model name")
    dataset_name: str = Field(..., description="Dataset the model was trained on")
    input_shape: List[int] = Field(..., description="Input image shape [height, width, channels]")
    num_classes: int = Field(..., description="Number of classes")
    class_names: List[str] = Field(..., description="List of class names")
    total_parameters: int = Field(..., description="Total number of model parameters")
    model_size_mb: float = Field(..., description="Model size in MB")
    training_date: Optional[datetime] = Field(None, description="Model training date")
    accuracy: Optional[float] = Field(None, description="Model accuracy")

class BatchClassificationRequest(BaseModel):
    """Request for batch image classification"""
    images_base64: List[str] = Field(..., description="List of base64 encoded images")
    model_name: ClassificationModel = Field(default=ClassificationModel.CNN_CUSTOM, description="Model to use for classification")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")
    top_k: int = Field(default=5, ge=1, le=10, description="Number of top predictions to return")

class BatchClassificationResponse(BaseModel):
    """Response for batch image classification"""
    batch_predictions: List[ImageClassificationResponse] = Field(..., description="Predictions for each image")
    total_images: int = Field(..., description="Total number of images processed")
    total_processing_time: float = Field(..., description="Total processing time in seconds")
    average_processing_time: float = Field(..., description="Average processing time per image")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    available_models: List[str] = Field(..., description="List of available models")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp")