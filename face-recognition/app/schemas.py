from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union
from datetime import datetime
import numpy as np

class FaceDetectionRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image containing faces")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence threshold for face detection")
    extract_faces: bool = Field(default=True, description="Whether to extract face images")
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
                "confidence_threshold": 0.5,
                "extract_faces": True
            }
        }

class FaceEmbedding(BaseModel):
    embedding: List[float] = Field(..., description="Face embedding vector (128-dimensional)")
    face_id: str = Field(..., description="Unique identifier for this face")
    
    class Config:
        json_schema_extra = {
            "example": {
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "face_id": "face_123"
            }
        }

class DetectedFace(BaseModel):
    face_id: str = Field(..., description="Unique identifier for this face")
    bbox: Dict[str, int] = Field(..., description="Bounding box coordinates {x, y, width, height}")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    face_image_base64: Optional[str] = Field(None, description="Base64 encoded face image (if extracted)")
    landmarks: Optional[Dict[str, List[int]]] = Field(None, description="Facial landmarks (eyes, nose, mouth)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "face_id": "face_123",
                "bbox": {"x": 100, "y": 150, "width": 80, "height": 80},
                "confidence": 0.95,
                "face_image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
                "landmarks": {
                    "left_eye": [110, 170],
                    "right_eye": [130, 170],
                    "nose": [120, 180],
                    "mouth_left": [115, 190],
                    "mouth_right": [125, 190]
                }
            }
        }

class FaceDetectionResponse(BaseModel):
    faces: List[DetectedFace] = Field(..., description="List of detected faces")
    total_faces: int = Field(..., description="Total number of faces detected")
    image_width: int = Field(..., description="Original image width")
    image_height: int = Field(..., description="Original image height")
    processing_time: float = Field(..., description="Processing time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "faces": [
                    {
                        "face_id": "face_123",
                        "bbox": {"x": 100, "y": 150, "width": 80, "height": 80},
                        "confidence": 0.95,
                        "face_image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
                    }
                ],
                "total_faces": 1,
                "image_width": 640,
                "image_height": 480,
                "processing_time": 0.5
            }
        }

class FaceRecognitionRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image of face to recognize")
    known_faces: List[FaceEmbedding] = Field(..., description="List of known face embeddings")
    similarity_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Similarity threshold for recognition")
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
                "known_faces": [
                    {
                        "embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        "face_id": "person_1"
                    }
                ],
                "similarity_threshold": 0.6
            }
        }

class FaceRecognitionResponse(BaseModel):
    recognized: bool = Field(..., description="Whether a matching face was found")
    matched_face_id: Optional[str] = Field(None, description="ID of the matched face (if recognized)")
    similarity_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Similarity score with matched face")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Recognition confidence")
    all_similarities: List[Dict[str, Union[str, float]]] = Field(..., description="Similarity scores with all known faces")
    
    class Config:
        json_schema_extra = {
            "example": {
                "recognized": True,
                "matched_face_id": "person_1",
                "similarity_score": 0.85,
                "confidence": 0.9,
                "all_similarities": [
                    {"face_id": "person_1", "similarity": 0.85},
                    {"face_id": "person_2", "similarity": 0.45}
                ]
            }
        }

class FaceComparisonRequest(BaseModel):
    image1_base64: str = Field(..., description="Base64 encoded first face image")
    image2_base64: str = Field(..., description="Base64 encoded second face image")
    
    class Config:
        json_schema_extra = {
            "example": {
                "image1_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
                "image2_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
            }
        }

class FaceComparisonResponse(BaseModel):
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score between the two faces")
    are_same_person: bool = Field(..., description="Whether the faces belong to the same person")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the comparison")
    
    class Config:
        json_schema_extra = {
            "example": {
                "similarity_score": 0.92,
                "are_same_person": True,
                "confidence": 0.95
            }
        }

class FaceRegistrationRequest(BaseModel):
    person_name: str = Field(..., description="Name of the person to register")
    face_images: List[str] = Field(..., description="List of base64 encoded face images for registration")
    
    class Config:
        json_schema_extra = {
            "example": {
                "person_name": "John Doe",
                "face_images": [
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
                ]
            }
        }

class FaceRegistrationResponse(BaseModel):
    person_id: str = Field(..., description="Unique ID for the registered person")
    person_name: str = Field(..., description="Name of the registered person")
    embeddings_created: int = Field(..., description="Number of face embeddings created")
    registration_status: str = Field(..., description="Status of registration")
    average_embedding: FaceEmbedding = Field(..., description="Average face embedding")
    
    class Config:
        json_schema_extra = {
            "example": {
                "person_id": "person_123",
                "person_name": "John Doe",
                "embeddings_created": 2,
                "registration_status": "success",
                "average_embedding": {
                    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    "face_id": "person_123_avg"
                }
            }
        }

class PersonInfo(BaseModel):
    person_id: str = Field(..., description="Unique ID for the person")
    person_name: str = Field(..., description="Name of the person")
    face_embedding: FaceEmbedding = Field(..., description="Face embedding")
    registration_date: str = Field(..., description="Date of registration")
    image_count: int = Field(..., description="Number of images used for registration")
    
    class Config:
        json_schema_extra = {
            "example": {
                "person_id": "person_123",
                "person_name": "John Doe",
                "face_embedding": {
                    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    "face_id": "person_123_avg"
                },
                "registration_date": "2024-01-15",
                "image_count": 3
            }
        }

class ModelInfo(BaseModel):
    model_name: str = Field(..., description="Name of the face recognition model")
    embedding_dimension: int = Field(..., description="Dimension of face embeddings")
    supported_image_formats: List[str] = Field(..., description="Supported image formats")
    max_image_size: int = Field(..., description="Maximum allowed image size in bytes")
    detection_confidence_threshold: float = Field(..., description="Default detection confidence threshold")
    recognition_threshold: float = Field(..., description="Default recognition similarity threshold")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "FaceNet",
                "embedding_dimension": 512,
                "supported_image_formats": ["jpg", "jpeg", "png", "bmp"],
                "max_image_size": 10485760,
                "detection_confidence_threshold": 0.5,
                "recognition_threshold": 0.6
            }
        }

class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    registered_persons: int = Field(..., description="Number of registered persons")
    models_loaded: bool = Field(..., description="Whether face recognition models are loaded")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00",
                "registered_persons": 10,
                "models_loaded": True
            }
        }