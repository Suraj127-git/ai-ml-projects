from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import logging

from .schemas import (
    TextToSQLRequest,
    TextToSQLResponse,
    SQLQuery,
    ModelInfo,
    HealthResponse,
    ModelName
)
from .model import TextToSQLModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Text-to-SQL API",
    description="API for converting natural language text to SQL queries",
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
model = TextToSQLModel()

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        model.load_model()
        logger.info("Text-to-SQL model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="Text-to-SQL API",
        version="1.0.0"
    )

@app.post("/generate-sql", response_model=TextToSQLResponse)
async def generate_sql(request: TextToSQLRequest):
    """Generate SQL query from natural language text"""
    try:
        sql_query = model.generate_sql(
            text=request.text,
            model_name=request.model_name,
            max_length=request.max_length,
            temperature=request.temperature
        )
        
        return TextToSQLResponse(
            text=request.text,
            sql_query=sql_query,
            model_name=request.model_name,
            confidence=0.95  # Placeholder confidence score
        )
    except Exception as e:
        logger.error(f"Error generating SQL: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models", response_model=List[ModelInfo])
async def get_models():
    """Get available models"""
    return [
        ModelInfo(
            name=ModelName.T5_SMALL,
            description="T5 Small model for text-to-SQL",
            parameters=60_000_000
        ),
        ModelInfo(
            name=ModelName.T5_BASE,
            description="T5 Base model for text-to-SQL",
            parameters=220_000_000
        ),
        ModelInfo(
            name=ModelName.T5_LARGE,
            description="T5 Large model for text-to-SQL",
            parameters=770_000_000
        ),
        ModelInfo(
            name=ModelName.FLAN_T5_SMALL,
            description="Flan-T5 Small model for text-to-SQL",
            parameters=80_000_000
        ),
        ModelInfo(
            name=ModelName.FLAN_T5_BASE,
            description="Flan-T5 Base model for text-to-SQL",
            parameters=250_000_000
        )
    ]

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {"status": "healthy", "model_loaded": model.is_loaded}