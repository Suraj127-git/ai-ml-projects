from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union
from datetime import datetime
from enum import Enum

class ProductCategory(str, Enum):
    """Product categories for classification"""
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    FOOD = "food"
    FURNITURE = "furniture"
    BOOKS = "books"
    TOYS = "toys"
    SPORTS = "sports"
    BEAUTY = "beauty"
    AUTOMOTIVE = "automotive"
    HOME = "home"
    JEWELRY = "jewelry"
    SHOES = "shoes"
    BAGS = "bags"
    WATCHES = "watches"
    HEALTH = "health"
    OFFICE = "office"
    GARDEN = "garden"
    PETS = "pets"
    BABY = "baby"
    TOOLS = "tools"

class ImageSize(str, Enum):
    """Supported image sizes"""
    SMALL = "224x224"
    MEDIUM = "299x299"
    LARGE = "384x384"

class ModelType(str, Enum):
    """Available model architectures"""
    EFFICIENTNET_B0 = "efficientnet_b0"
    EFFICIENTNET_B1 = "efficientnet_b1"
    EFFICIENTNET_B2 = "efficientnet_b2"
    RESNET50 = "resnet50"
    RESNET101 = "resnet101"
    MOBILENET_V2 = "mobilenet_v2"
    INCEPTION_V3 = "inception_v3"
    VGG16 = "vgg16"
    VGG19 = "vgg19"

class ClassificationRequest(BaseModel):
    """Request schema for image classification"""
    image_data: str = Field(..., description="Base64 encoded image data")
    model_type: ModelType = Field(default=ModelType.EFFICIENTNET_B0, description="Model architecture to use")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top predictions to return")
    confidence_threshold: float = Field(default=0.1, ge=0.0, le=1.0, description="Minimum confidence threshold")
    image_size: ImageSize = Field(default=ImageSize.SMALL, description="Image size for processing")
    
class ClassificationResponse(BaseModel):
    """Response schema for image classification"""
    predictions: List[Dict[str, Union[str, float]]] = Field(..., description="List of predictions with category and confidence")
    model_type: str = Field(..., description="Model used for classification")
    processing_time: float = Field(..., description="Processing time in seconds")
    image_size: str = Field(..., description="Processed image size")
    confidence_threshold: float = Field(..., description="Applied confidence threshold")
    timestamp: str = Field(..., description="Classification timestamp")
    
class BatchClassificationRequest(BaseModel):
    """Request schema for batch image classification"""
    images: List[str] = Field(..., description="List of base64 encoded image data", min_items=1, max_items=50)
    model_type: ModelType = Field(default=ModelType.EFFICIENTNET_B0, description="Model architecture to use")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top predictions per image")
    confidence_threshold: float = Field(default=0.1, ge=0.0, le=1.0, description="Minimum confidence threshold")
    image_size: ImageSize = Field(default=ImageSize.SMALL, description="Image size for processing")
    
class BatchClassificationResponse(BaseModel):
    """Response schema for batch image classification"""
    results: List[ClassificationResponse] = Field(..., description="List of classification results")
    total_images: int = Field(..., description="Total number of processed images")
    total_processing_time: float = Field(..., description="Total processing time in seconds")
    average_processing_time: float = Field(..., description="Average processing time per image")
    model_type: str = Field(..., description="Model used for classification")
    
class ModelTrainingRequest(BaseModel):
    """Request schema for model training"""
    dataset_path: str = Field(..., description="Path to training dataset")
    model_type: ModelType = Field(..., description="Model architecture to train")
    epochs: int = Field(default=10, ge=1, le=100, description="Number of training epochs")
    batch_size: int = Field(default=32, ge=8, le=256, description="Training batch size")
    learning_rate: float = Field(default=0.001, ge=0.0001, le=0.1, description="Learning rate")
    validation_split: float = Field(default=0.2, ge=0.1, le=0.4, description="Validation data split ratio")
    early_stopping_patience: int = Field(default=5, ge=1, le=20, description="Early stopping patience")
    
class ModelTrainingResponse(BaseModel):
    """Response schema for model training"""
    model_type: str = Field(..., description="Trained model architecture")
    epochs_trained: int = Field(..., description="Number of epochs actually trained")
    training_accuracy: float = Field(..., description="Final training accuracy")
    validation_accuracy: float = Field(..., description="Final validation accuracy")
    training_loss: float = Field(..., description="Final training loss")
    validation_loss: float = Field(..., description="Final validation loss")
    model_path: str = Field(..., description="Path to saved model")
    training_time: float = Field(..., description="Total training time in seconds")
    best_epoch: int = Field(..., description="Epoch with best validation accuracy")
    
class ModelInfo(BaseModel):
    """Model information schema"""
    model_type: str = Field(..., description="Model architecture")
    input_size: str = Field(..., description="Expected input image size")
    num_classes: int = Field(..., description="Number of output classes")
    parameters: int = Field(..., description="Number of model parameters")
    model_size_mb: float = Field(..., description="Model size in MB")
    training_date: Optional[str] = Field(None, description="Last training date")
    training_accuracy: Optional[float] = Field(None, description="Training accuracy")
    validation_accuracy: Optional[float] = Field(None, description="Validation accuracy")
    
class DatasetInfo(BaseModel):
    """Dataset information schema"""
    name: str = Field(..., description="Dataset name")
    num_classes: int = Field(..., description="Number of classes")
    num_images: int = Field(..., description="Total number of images")
    train_images: int = Field(..., description="Number of training images")
    validation_images: int = Field(..., description="Number of validation images")
    test_images: int = Field(..., description="Number of test images")
    class_distribution: Dict[str, int] = Field(..., description="Images per class")
    
class ProductInfo(BaseModel):
    """Product information schema"""
    name: str = Field(..., description="Product name")
    category: ProductCategory = Field(..., description="Product category")
    brand: Optional[str] = Field(None, description="Product brand")
    price_range: Optional[str] = Field(None, description="Price range")
    description: Optional[str] = Field(None, description="Product description")
    features: Optional[List[str]] = Field(None, description="Product features")
    
class ClassificationDetail(BaseModel):
    """Detailed classification information"""
    category: ProductCategory = Field(..., description="Predicted category")
    confidence: float = Field(..., description="Prediction confidence")
    probability: float = Field(..., description="Raw probability score")
    top_features: Optional[List[str]] = Field(None, description="Top contributing features")
    similar_products: Optional[List[ProductInfo]] = Field(None, description="Similar products")
    
class ModelComparisonRequest(BaseModel):
    """Request schema for model comparison"""
    image_data: str = Field(..., description="Base64 encoded image data")
    model_types: List[ModelType] = Field(..., description="List of models to compare", min_items=2, max_items=5)
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top predictions per model")
    confidence_threshold: float = Field(default=0.1, ge=0.0, le=1.0, description="Minimum confidence threshold")
    
class ModelComparisonResponse(BaseModel):
    """Response schema for model comparison"""
    comparison_results: Dict[str, ClassificationResponse] = Field(..., description="Results from each model")
    best_model: str = Field(..., description="Model with highest confidence")
    best_prediction: ClassificationDetail = Field(..., description="Best overall prediction")
    processing_times: Dict[str, float] = Field(..., description="Processing time for each model")
    consensus_category: Optional[str] = Field(None, description="Category agreed by majority of models")
    
class ImagePreprocessingRequest(BaseModel):
    """Request schema for image preprocessing"""
    image_data: str = Field(..., description="Base64 encoded image data")
    target_size: ImageSize = Field(default=ImageSize.SMALL, description="Target image size")
    normalize: bool = Field(default=True, description="Whether to normalize pixel values")
    augment: bool = Field(default=False, description="Whether to apply data augmentation")
    
class ImagePreprocessingResponse(BaseModel):
    """Response schema for image preprocessing"""
    processed_image: str = Field(..., description="Base64 encoded processed image")
    original_size: str = Field(..., description="Original image dimensions")
    processed_size: str = Field(..., description="Processed image dimensions")
    preprocessing_time: float = Field(..., description="Preprocessing time in seconds")
    augmentation_applied: bool = Field(..., description="Whether augmentation was applied")
    
class PerformanceMetrics(BaseModel):
    """Model performance metrics"""
    accuracy: float = Field(..., description="Overall accuracy")
    precision: float = Field(..., description="Average precision")
    recall: float = Field(..., description="Average recall")
    f1_score: float = Field(..., description="Average F1 score")
    top_3_accuracy: float = Field(..., description="Top-3 accuracy")
    top_5_accuracy: float = Field(..., description="Top-5 accuracy")
    confusion_matrix: Optional[List[List[int]]] = Field(None, description="Confusion matrix")
    classification_report: Optional[Dict[str, Dict[str, float]]] = Field(None, description="Detailed classification report")
    
class ModelEvaluationRequest(BaseModel):
    """Request schema for model evaluation"""
    model_type: ModelType = Field(..., description="Model to evaluate")
    test_dataset_path: str = Field(..., description="Path to test dataset")
    batch_size: int = Field(default=32, ge=8, le=256, description="Evaluation batch size")
    
class ModelEvaluationResponse(BaseModel):
    """Response schema for model evaluation"""
    model_type: str = Field(..., description="Evaluated model")
    performance_metrics: PerformanceMetrics = Field(..., description="Model performance metrics")
    evaluation_time: float = Field(..., description="Evaluation time in seconds")
    total_predictions: int = Field(..., description="Total number of predictions made")
    average_confidence: float = Field(..., description="Average prediction confidence")
    
class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    available_models: List[str] = Field(..., description="List of available models")
    loaded_models: List[str] = Field(..., description="List of currently loaded models")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    memory_usage: Dict[str, float] = Field(..., description="Memory usage statistics")
    timestamp: str = Field(..., description="Health check timestamp")