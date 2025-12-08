from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import Dict, Any, Optional, List
import uvicorn
import os

from .model import LeadScoringModel
from .schemas import (
    LeadData, LeadScore, BatchLeadScore, TrainingResponse, 
    ModelInfo, ModelPerformance, HealthResponse, SampleDataResponse
)

# Initialize FastAPI app
app = FastAPI(
    title="Lead Scoring System API",
    description="API for scoring leads using machine learning models to predict conversion probability",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
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
lead_scoring_model = LeadScoringModel()

# Load pre-trained models if available
models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, 'lead_scoring_models.pkl')

if os.path.exists(model_path):
    try:
        lead_scoring_model.load_models(model_path)
        print("✅ Pre-trained models loaded successfully")
    except Exception as e:
        print(f"⚠️ Could not load pre-trained models: {e}")
        print("Models will need to be trained first")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Lead Scoring System API",
        "version": "1.0.0",
        "status": "operational",
        "models_loaded": lead_scoring_model.is_trained
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        model_loaded=lead_scoring_model.is_trained,
        version="1.0.0"
    )

@app.post("/score", response_model=LeadScore)
async def score_lead(lead_data: LeadData, model_type: str = "random_forest"):
    """
    Score a single lead and predict conversion probability
    
    Args:
        lead_data: Lead information
        model_type: Model to use for scoring (xgboost, random_forest, logistic_regression)
    """
    try:
        if not lead_scoring_model.is_trained:
            raise HTTPException(
                status_code=503, 
                detail="Models not trained yet. Please train models first using /train endpoint."
            )
        
        # Convert Pydantic model to dict
        lead_dict = lead_data.model_dump()
        
        # Score the lead
        result = lead_scoring_model.predict_lead_score(lead_dict, model_type)
        
        # Add lead_id to result
        result['lead_id'] = lead_data.lead_id
        result['timestamp'] = datetime.now()
        
        return LeadScore(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scoring lead: {str(e)}")

@app.post("/score/batch", response_model=BatchLeadScore)
async def score_leads_batch(leads: List[LeadData], model_type: str = "random_forest"):
    """
    Score multiple leads in batch
    
    Args:
        leads: List of lead information
        model_type: Model to use for scoring
    """
    try:
        if not lead_scoring_model.is_trained:
            raise HTTPException(
                status_code=503, 
                detail="Models not trained yet. Please train models first using /train endpoint."
            )
        
        # Convert Pydantic models to dicts
        leads_dicts = [lead.model_dump() for lead in leads]
        
        # Score all leads
        start_time = datetime.now()
        results = lead_scoring_model.batch_predict(leads_dicts, model_type)
        end_time = datetime.now()
        
        # Convert to LeadScore objects
        lead_scores = []
        high_priority = 0
        medium_priority = 0
        low_priority = 0
        
        for result in results:
            lead_score = LeadScore(**result)
            lead_scores.append(lead_score)
            
            # Count priorities
            if lead_score.score >= 80:
                high_priority += 1
            elif lead_score.score >= 60:
                medium_priority += 1
            else:
                low_priority += 1
        
        processing_time = (end_time - start_time).total_seconds() * 1000  # milliseconds
        
        return BatchLeadScore(
            scores=lead_scores,
            total_processed=len(leads),
            high_priority=high_priority,
            medium_priority=medium_priority,
            low_priority=low_priority,
            processing_time_ms=processing_time
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scoring leads: {str(e)}")

@app.post("/train", response_model=TrainingResponse)
async def train_models(training_data: Optional[List[LeadData]] = None):
    """
    Train models on provided data or generate synthetic data
    
    Args:
        training_data: Optional training data. If not provided, synthetic data will be generated.
    """
    try:
        if training_data:
            # Use provided training data
            df = pd.DataFrame([lead.model_dump() for lead in training_data])
            print(f"Training on {len(df)} provided samples")
        else:
            # Generate synthetic training data
            print("Generating synthetic training data...")
            df = lead_scoring_model.generate_synthetic_training_data(n_samples=1000)
            print(f"Generated {len(df)} synthetic training samples")
        
        # Train models
        training_results = lead_scoring_model.train_models(df)
        
        # Select best model based on F1 score
        best_model = max(training_results.keys(), key=lambda k: training_results[k]['f1_score'] if training_results[k] else 0)
        best_results = training_results[best_model]
        
        # Save trained models
        lead_scoring_model.save_models(model_path)
        
        return TrainingResponse(
            message=f"Models trained successfully. Best model: {best_model}",
            model_type=best_model,
            accuracy=best_results['accuracy'],
            precision=best_results['precision'],
            recall=best_results['recall'],
            f1_score=best_results['f1_score'],
            auc_roc=best_results['auc_roc'],
            training_samples=len(df),
            training_date=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training models: {str(e)}")

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information"""
    try:
        return ModelInfo(**lead_scoring_model.get_model_info())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@app.get("/model/performance", response_model=Dict[str, ModelPerformance])
async def get_model_performance(model_type: Optional[str] = None):
    """Get detailed model performance metrics"""
    try:
        if not lead_scoring_model.is_trained:
            raise HTTPException(
                status_code=503, 
                detail="Models not trained yet. Please train models first."
            )
        
        performance = lead_scoring_model.get_model_performance(model_type)
        
        if 'error' in performance:
            raise HTTPException(status_code=404, detail=performance['error'])
        
        # Convert to ModelPerformance objects
        if model_type:
            return {model_type: ModelPerformance(**performance)}
        else:
            return {k: ModelPerformance(**v) for k, v in performance.items() if v is not None}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model performance: {str(e)}")

@app.post("/generate-sample-data", response_model=SampleDataResponse)
async def generate_sample_data(n_leads: int = 10):
    """
    Generate sample lead data for testing
    
    Args:
        n_leads: Number of sample leads to generate
    """
    try:
        # Generate synthetic lead data
        sample_leads_data = lead_scoring_model.generate_synthetic_lead_data(n_leads)
        
        # Convert to LeadData objects
        sample_leads = [LeadData(**lead_data) for lead_data in sample_leads_data]
        
        # Calculate summary statistics
        summary_stats = {
            "avg_engagement_score": sum(lead.engagement_score for lead in sample_leads) / len(sample_leads),
            "avg_website_visits": sum(lead.website_visits for lead in sample_leads) / len(sample_leads),
            "avg_email_opens": sum(lead.email_opens for lead in sample_leads) / len(sample_leads),
            "company_size_distribution": {},
            "industry_distribution": {},
            "job_title_distribution": {}
        }
        
        # Calculate distributions
        for lead in sample_leads:
            summary_stats["company_size_distribution"][lead.company_size] = summary_stats["company_size_distribution"].get(lead.company_size, 0) + 1
            summary_stats["industry_distribution"][lead.industry] = summary_stats["industry_distribution"].get(lead.industry, 0) + 1
            summary_stats["job_title_distribution"][lead.job_title] = summary_stats["job_title_distribution"].get(lead.job_title, 0) + 1
        
        return SampleDataResponse(
            sample_leads=sample_leads,
            total_generated=len(sample_leads),
            summary_stats=summary_stats,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating sample data: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)