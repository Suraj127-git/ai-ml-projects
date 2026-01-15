from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import logging
from datetime import datetime

from .model import recommendation_engine
from .schemas import (
    User, Item, Rating, RecommendationRequest, TrainingRequest,
    RecommendationResponse, TrainingResponse, EvaluationResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Collaborative Recommendation System API",
    description="API for collaborative filtering recommendations using matrix factorization",
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
    """Root endpoint with API information"""
    return {
        "message": "Collaborative Recommendation System API",
        "version": "1.0.0",
        "algorithms": ["matrix_factorization", "collaborative_filtering", "hybrid"],
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "engine_trained": recommendation_engine.is_trained,
        "total_users": len(recommendation_engine.users),
        "total_items": len(recommendation_engine.items),
        "total_ratings": len(recommendation_engine.ratings)
    }


@app.post("/users", response_model=dict)
async def add_user(user: User):
    """Add a new user to the system"""
    try:
        success = recommendation_engine.add_user(user)
        if success:
            return {"message": f"User {user.user_id} added successfully"}
        else:
            raise HTTPException(status_code=400, detail=f"User {user.user_id} already exists")
    except Exception as e:
        logger.error(f"Error adding user: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding user: {str(e)}")


@app.post("/items", response_model=dict)
async def add_item(item: Item):
    """Add a new item to the system"""
    try:
        success = recommendation_engine.add_item(item)
        if success:
            return {"message": f"Item {item.item_id} added successfully"}
        else:
            raise HTTPException(status_code=400, detail=f"Item {item.item_id} already exists")
    except Exception as e:
        logger.error(f"Error adding item: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding item: {str(e)}")


@app.post("/ratings", response_model=dict)
async def add_rating(rating: Rating):
    """Add a rating from a user to an item"""
    try:
        success = recommendation_engine.add_rating(rating)
        if success:
            return {"message": f"Rating added successfully"}
        else:
            raise HTTPException(status_code=400, detail="Invalid user or item ID")
    except Exception as e:
        logger.error(f"Error adding rating: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding rating: {str(e)}")


@app.post("/train", response_model=TrainingResponse)
async def train_model(training_request: TrainingRequest):
    """Train the recommendation model"""
    try:
        logger.info(f"Training request received: algorithm={training_request.algorithm}")
        
        if len(recommendation_engine.ratings) < 5:
            raise HTTPException(
                status_code=400, 
                detail="Insufficient data: at least 5 ratings required for training"
            )
        
        response = recommendation_engine.train(training_request)
        
        if response.status == "failed":
            raise HTTPException(status_code=500, detail=response.error_message)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Generate recommendations for a user"""
    try:
        if not recommendation_engine.is_trained:
            raise HTTPException(status_code=503, detail="Model not trained yet. Please train the model first.")
        
        logger.info(f"Recommendation request: user={request.user_id}, algorithm={request.algorithm}")
        
        response = recommendation_engine.generate_recommendations(request)
        
        if response.error_message:
            raise HTTPException(status_code=500, detail=response.error_message)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {str(e)}")


@app.get("/users/{user_id}/ratings")
async def get_user_ratings(user_id: str):
    """Get all ratings for a specific user"""
    try:
        if user_id not in recommendation_engine.users:
            raise HTTPException(status_code=404, detail=f'User {user_id} not found')
        
        user_ratings = [r for r in recommendation_engine.ratings if r.user_id == user_id]
        
        return {
            "user_id": user_id,
            "total_ratings": len(user_ratings),
            "ratings": [
                {
                    "item_id": r.item_id,
                    "rating": r.rating,
                    "timestamp": r.timestamp.isoformat() if hasattr(r, 'timestamp') else None
                }
                for r in user_ratings
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving user ratings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving ratings: {str(e)}")


@app.get("/items/{item_id}/ratings")
async def get_item_ratings(item_id: str):
    """Get all ratings for a specific item"""
    try:
        if item_id not in recommendation_engine.items:
            raise HTTPException(status_code=404, detail=f'Item {item_id} not found')
        
        item_ratings = [r for r in recommendation_engine.ratings if r.item_id == item_id]
        
        if not item_ratings:
            return {
                "item_id": item_id,
                "total_ratings": 0,
                "average_rating": 0.0,
                "ratings": []
            }
        
        avg_rating = sum(r.rating for r in item_ratings) / len(item_ratings)
        
        return {
            "item_id": item_id,
            "total_ratings": len(item_ratings),
            "average_rating": round(avg_rating, 2),
            "ratings": [
                {
                    "user_id": r.user_id,
                    "rating": r.rating,
                    "timestamp": r.timestamp.isoformat() if hasattr(r, 'timestamp') else None
                }
                for r in item_ratings
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving item ratings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving ratings: {str(e)}")


@app.post("/batch-ratings", response_model=dict)
async def add_batch_ratings(ratings: List[Rating]):
    """Add multiple ratings in batch"""
    try:
        success_count = 0
        failed_count = 0
        
        for rating in ratings:
            try:
                if recommendation_engine.add_rating(rating):
                    success_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                logger.warning(f"Failed to add rating for user {rating.user_id}, item {rating.item_id}: {str(e)}")
                failed_count += 1
        
        return {
            "message": f"Batch ratings processed",
            "success_count": success_count,
            "failed_count": failed_count,
            "total_count": len(ratings)
        }
        
    except Exception as e:
        logger.error(f"Error processing batch ratings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing batch ratings: {str(e)}")


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_model(test_ratings: List[Rating], k: int = 10):
    """Evaluate the recommendation model"""
    try:
        if not recommendation_engine.is_trained:
            raise HTTPException(status_code=503, detail="Model not trained yet. Please train the model first.")
        
        logger.info(f"Model evaluation request: {len(test_ratings)} test ratings, k={k}")
        
        response = recommendation_engine.evaluate_model(test_ratings, k)
        
        if response.error_message:
            raise HTTPException(status_code=500, detail=response.error_message)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model evaluation failed: {str(e)}")


@app.delete("/reset", response_model=dict)
async def reset_engine():
    """Reset the recommendation engine (clear all data)"""
    try:
        global recommendation_engine
        recommendation_engine = RecommendationEngine()
        
        return {
            "message": "Recommendation engine reset successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error resetting engine: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error resetting engine: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        # Calculate average ratings per user and item
        user_rating_counts = {}
        item_rating_counts = {}
        
        for rating in recommendation_engine.ratings:
            user_rating_counts[rating.user_id] = user_rating_counts.get(rating.user_id, 0) + 1
            item_rating_counts[rating.item_id] = item_rating_counts.get(rating.item_id, 0) + 1
        
        avg_ratings_per_user = np.mean(list(user_rating_counts.values())) if user_rating_counts else 0.0
        avg_ratings_per_item = np.mean(list(item_rating_counts.values())) if item_rating_counts else 0.0
        
        return {
            "total_users": len(recommendation_engine.users),
            "total_items": len(recommendation_engine.items),
            "total_ratings": len(recommendation_engine.ratings),
            "engine_trained": recommendation_engine.is_trained,
            "average_ratings_per_user": round(avg_ratings_per_user, 2),
            "average_ratings_per_item": round(avg_ratings_per_item, 2),
            "sparsity": 1.0 - (len(recommendation_engine.ratings) / (len(recommendation_engine.users) * len(recommendation_engine.items))) if recommendation_engine.users and recommendation_engine.items else 0.0
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)