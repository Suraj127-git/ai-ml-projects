from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
import base64
import json
import time
from datetime import datetime
import uvicorn

# Import custom modules
from .model import QualityControlModel
from .schemas import (
    QualityControlRequest, QualityControlResponse, BatchQualityRequest, 
    BatchQualityResponse, ModelInfo, QualityStatus, ProductCategory
)

# Create FastAPI app
app = FastAPI(
    title="Quality Control Computer Vision API",
    description="AI-powered quality control system using CNN/ResNet for defect detection",
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

# Initialize quality control model
quality_model = QualityControlModel()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Quality Control Computer Vision API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "inspect_product": "/inspect-product",
            "inspect_batch": "/inspect-batch",
            "upload_image": "/upload-image",
            "model_info": "/model-info",
            "quality_standards": "/quality-standards",
            "statistics": "/statistics"
        }
    }

@app.post("/inspect-product", response_model=QualityControlResponse)
async def inspect_product(request: QualityControlRequest):
    """
    Perform quality control inspection on a single product
    
    Args:
        request: Quality control request with product image and metadata
        
    Returns:
        Quality control inspection results
    """
    try:
        # Perform quality inspection
        result = quality_model.assess_quality(request)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality inspection failed: {str(e)}")

@app.post("/inspect-batch", response_model=BatchQualityResponse)
async def inspect_batch(request: BatchQualityRequest):
    """
    Perform quality control inspection on multiple products
    
    Args:
        request: Batch quality control request with multiple products
        
    Returns:
        Batch quality control results
    """
    try:
        # Perform batch inspection
        result = quality_model.batch_inspect(request)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inspection failed: {str(e)}")

@app.post("/upload-image")
async def upload_image(
    file: UploadFile = File(...),
    product_id: str = "unknown",
    product_name: str = "Unknown Product",
    category: ProductCategory = ProductCategory.ELECTRONICS,
    batch_id: Optional[str] = None
):
    """
    Upload and inspect a product image
    
    Args:
        file: Image file to upload
        product_id: Product identifier
        product_name: Product name
        category: Product category
        batch_id: Optional batch identifier
        
    Returns:
        Quality inspection results
    """
    try:
        # Read image file
        contents = await file.read()
        
        # Convert to base64
        image_base64 = base64.b64encode(contents).decode('utf-8')
        
        # Create inspection request
        request = QualityControlRequest(
            product_id=product_id,
            product_name=product_name,
            category=category,
            image_data=image_base64,
            image_format=file.content_type.split('/')[-1] if file.content_type else 'jpeg',
            batch_id=batch_id
        )
        
        # Perform inspection
        result = quality_model.assess_quality(request)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """
    Get information about the quality control model
    
    Returns:
        Model information including version, accuracy, and supported features
    """
    try:
        model_info = quality_model.get_model_info()
        
        return {
            "model_name": model_info.model_name,
            "version": model_info.version,
            "accuracy": model_info.accuracy,
            "training_date": model_info.training_date,
            "defect_types": [defect.value for defect in model_info.defect_types],
            "supported_categories": [cat.value for cat in model_info.categories],
            "processing_stats": quality_model.processing_stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@app.get("/quality-standards")
async def get_quality_standards(category: Optional[ProductCategory] = None):
    """
    Get quality standards for different product categories
    
    Args:
        category: Optional specific category to get standards for
        
    Returns:
        Quality standards configuration
    """
    try:
        if category:
            standards = quality_model.quality_standards.get(category)
            if not standards:
                raise HTTPException(status_code=404, detail=f'Standards not found for category: {category}')
            
            return {
                "category": standards.category.value,
                "max_defects": standards.max_defects,
                "min_quality_score": standards.min_quality_score,
                "critical_defect_threshold": standards.critical_defect_threshold,
                "dimensional_tolerance": standards.dimensional_tolerance,
                "color_variance_threshold": standards.color_variance_threshold
            }
        else:
            # Return all standards
            all_standards = {}
            for cat, standards in quality_model.quality_standards.items():
                all_standards[cat.value] = {
                    "max_defects": standards.max_defects,
                    "min_quality_score": standards.min_quality_score,
                    "critical_defect_threshold": standards.critical_defect_threshold,
                    "dimensional_tolerance": standards.dimensional_tolerance,
                    "color_variance_threshold": standards.color_variance_threshold
                }
            
            return all_standards
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get quality standards: {str(e)}")

@app.get("/statistics")
async def get_statistics():
    """
    Get quality control statistics
    
    Returns:
        Processing statistics and performance metrics
    """
    try:
        stats = quality_model.processing_stats
        
        return {
            "total_inspected": stats["total_inspected"],
            "defects_found": stats["defects_found"],
            "defect_rate": stats["defects_found"] / stats["total_inspected"] if stats["total_inspected"] > 0 else 0,
            "model_version": quality_model.model_version,
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@app.post("/generate-sample-data")
async def generate_sample_data(n_samples: int = 10):
    """
    Generate sample quality control data for testing
    
    Args:
        n_samples: Number of sample data points to generate
        
    Returns:
        Generated sample data
    """
    try:
        sample_data = quality_model.generate_synthetic_training_data(n_samples)
        
        return {
            "sample_data": sample_data,
            "count": len(sample_data),
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate sample data: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": True,
        "total_inspected": quality_model.processing_stats["total_inspected"],
        "defects_found": quality_model.processing_stats["defects_found"]
    }

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return {
        "error": "Invalid input",
        "detail": str(exc),
        "timestamp": datetime.now().isoformat()
    }

@app.exception_handler(RuntimeError)
async def runtime_error_handler(request, exc):
    return {
        "error": "Processing error",
        "detail": str(exc),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)