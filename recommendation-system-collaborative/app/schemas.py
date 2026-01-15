from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum


class RecommendationAlgorithm(str, Enum):
    MATRIX_FACTORIZATION = "matrix_factorization"
    COLLABORATIVE_FILTERING = "collaborative_filtering"
    HYBRID = "hybrid"
    CONTENT_BASED = "content_based"


class User(BaseModel):
    """User information for recommendation system"""
    user_id: str = Field(..., description="Unique user identifier")
    age: Optional[int] = Field(None, description="User age", ge=1, le=120)
    gender: Optional[str] = Field(None, description="User gender")
    location: Optional[str] = Field(None, description="User location")
    preferences: Optional[Dict] = Field(None, description="User preferences")
    registration_date: Optional[str] = Field(None, description="User registration date")


class Item(BaseModel):
    """Item information for recommendation system"""
    item_id: str = Field(..., description="Unique item identifier")
    name: str = Field(..., description="Item name")
    category: Optional[str] = Field(None, description="Item category")
    description: Optional[str] = Field(None, description="Item description")
    price: Optional[float] = Field(None, description="Item price", ge=0)
    features: Optional[Dict] = Field(None, description="Item features")
    tags: Optional[List[str]] = Field(None, description="Item tags")


class Rating(BaseModel):
    """User rating for an item"""
    user_id: str = Field(..., description="User ID")
    item_id: str = Field(..., description="Item ID")
    rating: float = Field(..., description="Rating value", ge=1, le=5)
    timestamp: Optional[str] = Field(None, description="Rating timestamp")
    context: Optional[Dict] = Field(None, description="Rating context")


class Interaction(BaseModel):
    """User interaction with an item"""
    user_id: str = Field(..., description="User ID")
    item_id: str = Field(..., description="Item ID")
    interaction_type: str = Field(..., description="Type of interaction (view, click, purchase, etc.)")
    timestamp: str = Field(..., description="Interaction timestamp")
    duration: Optional[float] = Field(None, description="Interaction duration in seconds")
    context: Optional[Dict] = Field(None, description="Interaction context")


class RecommendationRequest(BaseModel):
    """Request for recommendations"""
    user_id: str = Field(..., description="User ID to generate recommendations for")
    algorithm: RecommendationAlgorithm = Field(
        default=RecommendationAlgorithm.MATRIX_FACTORIZATION,
        description="Recommendation algorithm to use"
    )
    num_recommendations: int = Field(default=10, ge=1, le=50, description="Number of recommendations")
    categories: Optional[List[str]] = Field(None, description="Filter by categories")
    exclude_rated: bool = Field(default=True, description="Exclude already rated items")
    diversity_weight: float = Field(default=0.1, ge=0, le=1, description="Diversity weight for recommendations")


class RecommendationResponse(BaseModel):
    """Response with recommendations"""
    user_id: str = Field(..., description="User ID")
    recommendations: List[Dict] = Field(..., description="List of recommended items with scores")
    algorithm: str = Field(..., description="Algorithm used")
    num_recommendations: int = Field(..., description="Number of recommendations requested")
    execution_time: float = Field(..., description="Execution time in seconds")
    diversity_score: Optional[float] = Field(None, description="Diversity score of recommendations")


class TrainingRequest(BaseModel):
    """Request to train recommendation models"""
    algorithm: RecommendationAlgorithm = Field(
        default=RecommendationAlgorithm.MATRIX_FACTORIZATION,
        description="Algorithm to train"
    )
    min_ratings: int = Field(default=5, ge=1, description="Minimum ratings per user/item")
    test_split: float = Field(default=0.2, ge=0.1, le=0.5, description="Test split ratio")
    validation_split: float = Field(default=0.1, ge=0.05, le=0.3, description="Validation split ratio")
    hyperparameters: Optional[Dict] = Field(None, description="Model hyperparameters")


class TrainingResponse(BaseModel):
    """Response from model training"""
    algorithm: str = Field(..., description="Algorithm trained")
    training_samples: int = Field(..., description="Number of training samples")
    test_samples: int = Field(..., description="Number of test samples")
    metrics: Dict = Field(..., description="Training metrics")
    hyperparameters: Dict = Field(..., description="Used hyperparameters")
    training_time: float = Field(..., description="Training time in seconds")
    model_size: Optional[str] = Field(None, description="Model size information")


class SimilarUsersRequest(BaseModel):
    """Request to find similar users"""
    user_id: str = Field(..., description="User ID to find similar users for")
    num_similar: int = Field(default=10, ge=1, le=50, description="Number of similar users")
    similarity_threshold: float = Field(default=0.5, ge=0, le=1, description="Minimum similarity threshold")


class SimilarUsersResponse(BaseModel):
    """Response with similar users"""
    user_id: str = Field(..., description="User ID")
    similar_users: List[Dict] = Field(..., description="List of similar users with similarity scores")
    algorithm: str = Field(..., description="Algorithm used")


class SimilarItemsRequest(BaseModel):
    """Request to find similar items"""
    item_id: str = Field(..., description="Item ID to find similar items for")
    num_similar: int = Field(default=10, ge=1, le=50, description="Number of similar items")
    similarity_threshold: float = Field(default=0.3, ge=0, le=1, description="Minimum similarity threshold")
    categories: Optional[List[str]] = Field(None, description="Filter by categories")


class SimilarItemsResponse(BaseModel):
    """Response with similar items"""
    item_id: str = Field(..., description="Item ID")
    similar_items: List[Dict] = Field(..., description="List of similar items with similarity scores")
    algorithm: str = Field(..., description="Algorithm used")


class ModelInfo(BaseModel):
    """Information about a recommendation model"""
    algorithm: str = Field(..., description="Algorithm name")
    training_date: str = Field(..., description="Model training date")
    model_size: Optional[str] = Field(None, description="Model size")
    hyperparameters: Optional[Dict] = Field(None, description="Model hyperparameters")
    performance_metrics: Optional[Dict] = Field(None, description="Model performance metrics")
    num_users: Optional[int] = Field(None, description="Number of users in training data")
    num_items: Optional[int] = Field(None, description="Number of items in training data")
    num_ratings: Optional[int] = Field(None, description="Number of ratings in training data")


class EvaluationRequest(BaseModel):
    """Request to evaluate recommendation model"""
    algorithm: RecommendationAlgorithm = Field(
        default=RecommendationAlgorithm.MATRIX_FACTORIZATION,
        description="Algorithm to evaluate"
    )
    test_users: Optional[List[str]] = Field(None, description="Specific users to evaluate")
    k: int = Field(default=10, ge=1, le=50, description="Number of recommendations to evaluate")
    metrics: Optional[List[str]] = Field(None, description="Specific metrics to calculate")


class EvaluationResponse(BaseModel):
    """Response with evaluation results"""
    algorithm: str = Field(..., description="Algorithm evaluated")
    precision_at_k: float = Field(..., description="Precision at k")
    recall_at_k: float = Field(..., description="Recall at k")
    f1_score_at_k: float = Field(..., description="F1 score at k")
    ndcg_at_k: float = Field(..., description="NDCG at k")
    map_score: float = Field(..., description="Mean Average Precision")
    coverage: float = Field(..., description="Catalog coverage")
    diversity: float = Field(..., description="Recommendation diversity")
    novelty: float = Field(..., description="Recommendation novelty")
    evaluation_samples: int = Field(..., description="Number of evaluation samples")
    evaluation_time: float = Field(..., description="Evaluation time in seconds")


class DataUploadRequest(BaseModel):
    """Request to upload recommendation data"""
    data_type: str = Field(..., description="Type of data: ratings, interactions, users, items")
    format: str = Field(default="json", description="Data format: json, csv")
    overwrite: bool = Field(default=False, description="Whether to overwrite existing data")


class DataUploadResponse(BaseModel):
    """Response from data upload"""
    data_type: str = Field(..., description="Type of data uploaded")
    records_processed: int = Field(..., description="Number of records processed")
    records_inserted: int = Field(..., description="Number of records inserted")
    errors: Optional[List[str]] = Field(None, description="Processing errors")
    upload_time: float = Field(..., description="Upload time in seconds")