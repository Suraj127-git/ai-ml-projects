import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import logging

from .schemas import (
    User, Item, Rating, RecommendationRequest, TrainingRequest,
    RecommendationResponse, TrainingResponse, EvaluationResponse
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    Collaborative filtering recommendation engine with multiple algorithms:
    - Matrix Factorization (NMF)
    - User-based Collaborative Filtering
    - Item-based Collaborative Filtering
    - Hybrid approaches
    """
    
    def __init__(self):
        self.user_item_matrix = None
        self.user_features = None
        self.item_features = None
        self.user_similarity = None
        self.item_similarity = None
        self.users = {}
        self.items = {}
        self.ratings = []
        self.nmf_model = None
        self.is_trained = False
        self.user_encoder = {}
        self.item_encoder = {}
        self.user_decoder = {}
        self.item_decoder = {}
        
    def add_user(self, user: User) -> bool:
        """Add a new user to the system"""
        try:
            if user.user_id not in self.users:
                self.users[user.user_id] = user
                logger.info(f"Added user: {user.user_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error adding user {user.user_id}: {str(e)}")
            return False
    
    def add_item(self, item: Item) -> bool:
        """Add a new item to the system"""
        try:
            if item.item_id not in self.items:
                self.items[item.item_id] = item
                logger.info(f"Added item: {item.item_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error adding item {item.item_id}: {str(e)}")
            return False
    
    def add_rating(self, rating: Rating) -> bool:
        """Add a rating from a user to an item"""
        try:
            if rating.user_id not in self.users or rating.item_id not in self.items:
                logger.warning(f"User {rating.user_id} or item {rating.item_id} not found")
                return False
            
            self.ratings.append(rating)
            logger.info(f"Added rating: user={rating.user_id}, item={rating.item_id}, rating={rating.rating}")
            return True
        except Exception as e:
            logger.error(f"Error adding rating: {str(e)}")
            return False
    
    def _build_user_item_matrix(self) -> np.ndarray:
        """Build user-item interaction matrix from ratings"""
        try:
            # Create encoders for users and items
            unique_users = sorted(list(set(r.user_id for r in self.ratings)))
            unique_items = sorted(list(set(r.item_id for r in self.ratings)))
            
            self.user_encoder = {user_id: idx for idx, user_id in enumerate(unique_users)}
            self.item_encoder = {item_id: idx for idx, item_id in enumerate(unique_items)}
            self.user_decoder = {idx: user_id for user_id, idx in self.user_encoder.items()}
            self.item_decoder = {idx: item_id for item_id, idx in self.item_encoder.items()}
            
            # Build matrix
            n_users = len(unique_users)
            n_items = len(unique_items)
            matrix = np.zeros((n_users, n_items))
            
            for rating in self.ratings:
                user_idx = self.user_encoder[rating.user_id]
                item_idx = self.item_encoder[rating.item_id]
                matrix[user_idx, item_idx] = rating.rating
            
            return matrix
        except Exception as e:
            logger.error(f"Error building user-item matrix: {str(e)}")
            raise
    
    def _matrix_factorization_nmf(self, n_components: int = 50, max_iter: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Perform Non-negative Matrix Factorization"""
        try:
            if self.user_item_matrix is None or self.user_item_matrix.size == 0:
                raise ValueError("User-item matrix is empty")
            
            # Handle missing values (0s) by using a small positive value
            matrix_filled = self.user_item_matrix.copy()
            matrix_filled[matrix_filled == 0] = 0.1
            
            nmf = NMF(n_components=n_components, max_iter=max_iter, random_state=42)
            user_features = nmf.fit_transform(matrix_filled)
            item_features = nmf.components_.T
            
            logger.info(f"NMF completed: user_features shape={user_features.shape}, item_features shape={item_features.shape}")
            return user_features, item_features
        except Exception as e:
            logger.error(f"Error in NMF matrix factorization: {str(e)}")
            raise
    
    def _calculate_similarities(self):
        """Calculate user-user and item-item similarities"""
        try:
            if self.user_features is not None:
                self.user_similarity = cosine_similarity(self.user_features)
            
            if self.item_features is not None:
                self.item_similarity = cosine_similarity(self.item_features)
            
            logger.info("Similarity matrices calculated")
        except Exception as e:
            logger.error(f"Error calculating similarities: {str(e)}")
            raise
    
    def _predict_rating_nmf(self, user_id: str, item_id: str) -> float:
        """Predict rating using matrix factorization"""
        try:
            if user_id not in self.user_encoder or item_id not in self.item_encoder:
                return 0.0
            
            user_idx = self.user_encoder[user_id]
            item_idx = self.item_encoder[item_id]
            
            if self.user_features is not None and self.item_features is not None:
                prediction = np.dot(self.user_features[user_idx], self.item_features[item_idx])
                return float(np.clip(prediction, 1.0, 5.0))
            
            return 0.0
        except Exception as e:
            logger.error(f"Error predicting rating for user {user_id}, item {item_id}: {str(e)}")
            return 0.0
    
    def _collaborative_filtering_user_based(self, user_id: str, item_id: str, k: int = 10) -> float:
        """Predict rating using user-based collaborative filtering"""
        try:
            if user_id not in self.user_encoder or item_id not in self.item_encoder:
                return 0.0
            
            user_idx = self.user_encoder[user_id]
            item_idx = self.item_encoder[item_id]
            
            if self.user_similarity is None:
                return 0.0
            
            # Find k most similar users
            similar_users = np.argsort(self.user_similarity[user_idx])[::-1][1:k+1]
            
            # Calculate weighted average of ratings from similar users
            numerator = 0.0
            denominator = 0.0
            
            for similar_user_idx in similar_users:
                if self.user_item_matrix[similar_user_idx, item_idx] > 0:
                    similarity = self.user_similarity[user_idx, similar_user_idx]
                    rating = self.user_item_matrix[similar_user_idx, item_idx]
                    numerator += similarity * rating
                    denominator += abs(similarity)
            
            if denominator > 0:
                return float(np.clip(numerator / denominator, 1.0, 5.0))
            
            return 0.0
        except Exception as e:
            logger.error(f"Error in user-based CF for user {user_id}, item {item_id}: {str(e)}")
            return 0.0
    
    def _collaborative_filtering_item_based(self, user_id: str, item_id: str, k: int = 10) -> float:
        """Predict rating using item-based collaborative filtering"""
        try:
            if user_id not in self.user_encoder or item_id not in self.item_encoder:
                return 0.0
            
            user_idx = self.user_encoder[user_id]
            item_idx = self.item_encoder[item_id]
            
            if self.item_similarity is None:
                return 0.0
            
            # Find k most similar items
            similar_items = np.argsort(self.item_similarity[item_idx])[::-1][1:k+1]
            
            # Calculate weighted average of user's ratings for similar items
            numerator = 0.0
            denominator = 0.0
            
            for similar_item_idx in similar_items:
                if self.user_item_matrix[user_idx, similar_item_idx] > 0:
                    similarity = self.item_similarity[item_idx, similar_item_idx]
                    rating = self.user_item_matrix[user_idx, similar_item_idx]
                    numerator += similarity * rating
                    denominator += abs(similarity)
            
            if denominator > 0:
                return float(np.clip(numerator / denominator, 1.0, 5.0))
            
            return 0.0
        except Exception as e:
            logger.error(f"Error in item-based CF for user {user_id}, item {item_id}: {str(e)}")
            return 0.0
    
    def train(self, request: TrainingRequest) -> TrainingResponse:
        """Train the recommendation engine with specified algorithm"""
        try:
            logger.info(f"Starting training with algorithm: {request.algorithm}")
            start_time = datetime.now()
            
            if len(self.ratings) < 10:
                raise ValueError("Insufficient ratings data (minimum 10 required)")
            
            # Build user-item matrix
            self.user_item_matrix = self._build_user_item_matrix()
            
            # Train based on algorithm
            if request.algorithm == "matrix_factorization":
                self.user_features, self.item_features = self._matrix_factorization_nmf(
                    n_components=request.hyperparameters.get("n_components", 50),
                    max_iter=request.hyperparameters.get("max_iter", 200)
                )
            elif request.algorithm == "collaborative_filtering":
                # For collaborative filtering, we still use matrix factorization as base
                self.user_features, self.item_features = self._matrix_factorization_nmf(
                    n_components=request.hyperparameters.get("n_components", 50),
                    max_iter=request.hyperparameters.get("max_iter", 200)
                )
            elif request.algorithm == "hybrid":
                # Combine multiple approaches
                self.user_features, self.item_features = self._matrix_factorization_nmf(
                    n_components=request.hyperparameters.get("n_components", 50),
                    max_iter=request.hyperparameters.get("max_iter", 200)
                )
            else:
                raise ValueError(f"Unsupported algorithm: {request.algorithm}")
            
            # Calculate similarities
            self._calculate_similarities()
            
            self.is_trained = True
            training_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Training completed successfully in {training_time:.2f} seconds")
            
            return TrainingResponse(
                status="success",
                algorithm=request.algorithm,
                training_samples=len(self.ratings),
                training_time_seconds=training_time,
                model_parameters={
                    "n_users": len(self.user_encoder),
                    "n_items": len(self.item_encoder),
                    "n_components": request.hyperparameters.get("n_components", 50)
                }
            )
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return TrainingResponse(
                status="failed",
                algorithm=request.algorithm,
                training_samples=len(self.ratings),
                training_time_seconds=0.0,
                error_message=str(e)
            )
    
    def generate_recommendations(self, request: RecommendationRequest) -> RecommendationResponse:
        """Generate recommendations for a user"""
        try:
            if not self.is_trained:
                raise ValueError("Model not trained yet")
            
            if request.user_id not in self.users:
                raise ValueError(f"User {request.user_id} not found")
            
            user_id = request.user_id
            num_recommendations = request.num_recommendations
            algorithm = request.algorithm
            
            # Get items the user hasn't interacted with
            user_rated_items = set(r.item_id for r in self.ratings if r.user_id == user_id)
            candidate_items = [item_id for item_id in self.items.keys() if item_id not in user_rated_items]
            
            if not candidate_items:
                # If user has rated all items, recommend based on highest average ratings
                candidate_items = list(self.items.keys())
            
            # Generate predictions
            predictions = []
            for item_id in candidate_items:
                if algorithm == "matrix_factorization":
                    score = self._predict_rating_nmf(user_id, item_id)
                elif algorithm == "collaborative_filtering":
                    # Use user-based collaborative filtering
                    score = self._collaborative_filtering_user_based(user_id, item_id)
                elif algorithm == "item_based":
                    score = self._collaborative_filtering_item_based(user_id, item_id)
                elif algorithm == "hybrid":
                    # Combine multiple methods
                    nmf_score = self._predict_rating_nmf(user_id, item_id)
                    user_cf_score = self._collaborative_filtering_user_based(user_id, item_id)
                    score = 0.6 * nmf_score + 0.4 * user_cf_score
                else:
                    score = self._predict_rating_nmf(user_id, item_id)
                
                predictions.append((item_id, score))
            
            # Sort by predicted score and return top N
            predictions.sort(key=lambda x: x[1], reverse=True)
            top_predictions = predictions[:num_recommendations]
            
            recommendations = []
            for item_id, score in top_predictions:
                item = self.items[item_id]
                recommendations.append({
                    "item_id": item_id,
                    "item_name": item.item_name,
                    "category": item.category,
                    "predicted_rating": round(score, 2),
                    "confidence": min(abs(score - 3.0) / 2.0, 1.0)  # Simple confidence metric
                })
            
            return RecommendationResponse(
                user_id=user_id,
                recommendations=recommendations,
                algorithm=algorithm,
                total_candidates=len(candidate_items)
            )
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {str(e)}")
            return RecommendationResponse(
                user_id=request.user_id,
                recommendations=[],
                algorithm=request.algorithm,
                total_candidates=0,
                error_message=str(e)
            )
    
    def evaluate_model(self, test_ratings: List[Rating], k: int = 10) -> EvaluationResponse:
        """Evaluate the recommendation model"""
        try:
            if not self.is_trained:
                raise ValueError("Model not trained yet")
            
            if len(test_ratings) == 0:
                raise ValueError("No test ratings provided")
            
            # Group test ratings by user
            user_test_ratings = {}
            for rating in test_ratings:
                if rating.user_id not in user_test_ratings:
                    user_test_ratings[rating.user_id] = []
                user_test_ratings[rating.user_id].append(rating)
            
            precision_scores = []
            recall_scores = []
            
            for user_id, user_ratings in user_test_ratings.items():
                if user_id not in self.users:
                    continue
                
                # Get top k items from test set
                test_items = set(r.item_id for r in user_ratings)
                if len(test_items) == 0:
                    continue
                
                # Generate recommendations for this user
                request = RecommendationRequest(
                    user_id=user_id,
                    num_recommendations=k,
                    algorithm="matrix_factorization"
                )
                
                response = self.generate_recommendations(request)
                if response.error_message:
                    continue
                
                recommended_items = set(r["item_id"] for r in response.recommendations)
                
                # Calculate precision and recall
                relevant_recommended = len(recommended_items.intersection(test_items))
                
                precision = relevant_recommended / len(recommended_items) if len(recommended_items) > 0 else 0.0
                recall = relevant_recommended / len(test_items) if len(test_items) > 0 else 0.0
                
                precision_scores.append(precision)
                recall_scores.append(recall)
            
            avg_precision = np.mean(precision_scores) if precision_scores else 0.0
            avg_recall = np.mean(recall_scores) if recall_scores else 0.0
            f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
            
            return EvaluationResponse(
                precision_at_k=avg_precision,
                recall_at_k=avg_recall,
                f1_score=f1_score,
                num_test_users=len(user_test_ratings),
                k=k
            )
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            return EvaluationResponse(
                precision_at_k=0.0,
                recall_at_k=0.0,
                f1_score=0.0,
                num_test_users=0,
                k=k,
                error_message=str(e)
            )


# Global recommendation engine instance
recommendation_engine = RecommendationEngine()