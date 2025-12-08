from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class DefectType(str, Enum):
    """Types of defects that can be detected"""
    SCRATCH = "scratch"
    CRACK = "crack"
    DENT = "dent"
    DISCOLORATION = "discoloration"
    MISSING_PART = "missing_part"
    DIMENSIONAL_ERROR = "dimensional_error"
    SURFACE_DEFECT = "surface_defect"
    CONTAMINATION = "contamination"

class ProductCategory(str, Enum):
    """Product categories for quality control"""
    ELECTRONICS = "electronics"
    AUTOMOTIVE = "automotive"
    TEXTILES = "textiles"
    FOOD = "food"
    PHARMACEUTICAL = "pharmaceutical"
    METAL = "metal"
    PLASTIC = "plastic"
    CERAMIC = "ceramic"

class QualityStatus(str, Enum):
    """Quality assessment status"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    NEEDS_REVIEW = "needs_review"

class QualityControlRequest(BaseModel):
    """Request schema for quality control inspection"""
    product_id: str = Field(..., description="Unique product identifier")
    product_name: str = Field(..., description="Product name")
    category: ProductCategory = Field(..., description="Product category")
    image_data: str = Field(..., description="Base64 encoded product image")
    image_format: str = Field(default="jpeg", description="Image format (jpeg, png, etc.)")
    quality_standards: Dict[str, float] = Field(default_factory=dict, description="Quality thresholds")
    inspection_type: str = Field(default="automated", description="Type of inspection")
    batch_id: Optional[str] = Field(None, description="Batch identifier for batch processing")
    manufacturing_date: Optional[datetime] = Field(None, description="Product manufacturing date")

class DefectDetection(BaseModel):
    """Individual defect detection result"""
    defect_type: DefectType = Field(..., description="Type of defect detected")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    location: Dict[str, int] = Field(..., description="Defect location (x, y, width, height)")
    severity: float = Field(..., ge=0.0, le=1.0, description="Defect severity score")
    description: Optional[str] = Field(None, description="Detailed defect description")

class QualityMetrics(BaseModel):
    """Quality assessment metrics"""
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score")
    defect_count: int = Field(..., ge=0, description="Total number of defects detected")
    critical_defects: int = Field(..., ge=0, description="Number of critical defects")
    minor_defects: int = Field(..., ge=0, description="Number of minor defects")
    dimensional_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0, description="Dimensional accuracy score")
    surface_quality: Optional[float] = Field(None, ge=0.0, le=1.0, description="Surface quality score")
    color_consistency: Optional[float] = Field(None, ge=0.0, le=1.0, description="Color consistency score")

class QualityControlResponse(BaseModel):
    """Response schema for quality control inspection"""
    product_id: str = Field(..., description="Product identifier")
    inspection_id: str = Field(..., description="Unique inspection identifier")
    status: QualityStatus = Field(..., description="Overall quality status")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score")
    defects_detected: List[DefectDetection] = Field(..., description="List of detected defects")
    metrics: QualityMetrics = Field(..., description="Detailed quality metrics")
    processing_time: float = Field(..., description="Processing time in seconds")
    recommendations: List[str] = Field(default_factory=list, description="Quality improvement recommendations")
    inspection_timestamp: datetime = Field(..., description="Inspection timestamp")
    model_version: str = Field(..., description="Model version used for inspection")

class BatchQualityRequest(BaseModel):
    """Request schema for batch quality control"""
    products: List[QualityControlRequest] = Field(..., description="List of products to inspect")
    batch_config: Dict[str, Any] = Field(default_factory=dict, description="Batch processing configuration")

class BatchQualityResponse(BaseModel):
    """Response schema for batch quality control"""
    batch_id: str = Field(..., description="Batch identifier")
    total_products: int = Field(..., description="Total number of products processed")
    passed_products: int = Field(..., description="Number of products that passed inspection")
    failed_products: int = Field(..., description="Number of products that failed inspection")
    warning_products: int = Field(..., description="Number of products with warnings")
    processing_time: float = Field(..., description="Total processing time in seconds")
    individual_results: List[QualityControlResponse] = Field(..., description="Individual product results")
    batch_summary: Dict[str, Any] = Field(..., description="Batch-level summary statistics")

class ModelInfo(BaseModel):
    """Model information schema"""
    model_name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    accuracy: float = Field(..., description="Model accuracy on validation set")
    training_date: datetime = Field(..., description="Model training date")
    defect_types: List[DefectType] = Field(..., description="Supported defect types")
    categories: List[ProductCategory] = Field(..., description="Supported product categories")

class QualityStandards(BaseModel):
    """Quality standards configuration"""
    category: ProductCategory = Field(..., description="Product category")
    max_defects: int = Field(..., description="Maximum allowed defects per product")
    min_quality_score: float = Field(..., ge=0.0, le=1.0, description="Minimum quality score")
    critical_defect_threshold: float = Field(..., ge=0.0, le=1.0, description="Critical defect confidence threshold")
    dimensional_tolerance: Optional[float] = Field(None, description="Dimensional tolerance percentage")
    color_variance_threshold: Optional[float] = Field(None, description="Color variance threshold")

class TrainingData(BaseModel):
    """Training data schema"""
    image_data: str = Field(..., description="Base64 encoded training image")
    defect_annotations: List[DefectDetection] = Field(..., description="Defect annotations")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Quality score")
    category: ProductCategory = Field(..., description="Product category")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")