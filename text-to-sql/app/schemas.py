from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union
from datetime import datetime
from enum import Enum


class ModelName(str, Enum):
    T5_SMALL = "t5-small"
    T5_BASE = "t5-base"
    T5_LARGE = "t5-large"
    FLAN_T5_SMALL = "flan-t5-small"
    FLAN_T5_BASE = "flan-t5-base"
    FLAN_T5_LARGE = "flan-t5-large"
    BART_BASE = "bart-base"
    BART_LARGE = "bart-large"
    CODE_T5_SMALL = "codet5-small"
    CODE_T5_BASE = "codet5-base"


class DatabaseType(str, Enum):
    SQLITE = "sqlite"
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    MSSQL = "mssql"
    ORACLE = "oracle"


class SQLGenerationMode(str, Enum):
    STANDARD = "standard"
    STRICT = "strict"
    FLEXIBLE = "flexible"


class QueryType(str, Enum):
    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    CREATE = "CREATE"
    ALTER = "ALTER"
    DROP = "DROP"
    UNKNOWN = "UNKNOWN"


class ValidationLevel(str, Enum):
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"


# Request Schemas
class TextToSQLRequest(BaseModel):
    natural_language_query: str = Field(..., description="Natural language query to convert to SQL")
    database_schema: Optional[Dict[str, List[Dict[str, str]]]] = Field(
        None, 
        description="Database schema information with table names and column details"
    )
    database_type: DatabaseType = Field(default=DatabaseType.SQLITE, description="Target database type")
    model_name: ModelName = Field(default=ModelName.T5_BASE, description="Model to use for SQL generation")
    context: Optional[str] = Field(None, description="Additional context about the database or domain")
    examples: Optional[List[Dict[str, str]]] = Field(
        None,
        description="Few-shot examples of natural language to SQL pairs"
    )
    generation_mode: SQLGenerationMode = Field(
        default=SQLGenerationMode.STANDARD,
        description="SQL generation mode (strict/flexible)"
    )
    validation_level: ValidationLevel = Field(
        default=ValidationLevel.BASIC,
        description="SQL validation level"
    )
    max_length: int = Field(default=512, description="Maximum length of generated SQL")
    temperature: float = Field(default=0.1, description="Generation temperature (0.0-1.0)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "natural_language_query": "Show me all customers who made purchases in the last month",
                "database_schema": {
                    "customers": [
                        {"name": "customer_id", "type": "INTEGER", "description": "Primary key"},
                        {"name": "name", "type": "TEXT", "description": "Customer name"},
                        {"name": "email", "type": "TEXT", "description": "Customer email"}
                    ],
                    "orders": [
                        {"name": "order_id", "type": "INTEGER", "description": "Primary key"},
                        {"name": "customer_id", "type": "INTEGER", "description": "Foreign key to customers"},
                        {"name": "order_date", "type": "DATE", "description": "Order date"},
                        {"name": "total_amount", "type": "DECIMAL", "description": "Total order amount"}
                    ]
                },
                "database_type": "sqlite",
                "model_name": "t5-base",
                "context": "E-commerce database with customer and order information",
                "generation_mode": "standard",
                "validation_level": "basic"
            }
        }


class BatchTextToSQLRequest(BaseModel):
    queries: List[TextToSQLRequest] = Field(..., description="List of text-to-SQL queries")
    use_same_schema: bool = Field(default=True, description="Use same schema for all queries")
    
    class Config:
        json_schema_extra = {
            "example": {
                "queries": [
                    {
                        "natural_language_query": "Show all customers",
                        "database_type": "sqlite"
                    },
                    {
                        "natural_language_query": "Count total orders",
                        "database_type": "sqlite"
                    }
                ],
                "use_same_schema": True
            }
        }


# Response Schemas
class SQLGenerationResult(BaseModel):
    sql_query: str = Field(..., description="Generated SQL query")
    query_type: QueryType = Field(..., description="Type of SQL query")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")
    execution_time: float = Field(..., description="Generation time in seconds")
    model_used: str = Field(..., description="Model used for generation")
    
    
class ValidationResult(BaseModel):
    is_valid: bool = Field(..., description="Whether the SQL is syntactically valid")
    errors: List[str] = Field(default_factory=list, description="List of validation errors")
    warnings: List[str] = Field(default_factory=list, description="List of validation warnings")
    suggestions: List[str] = Field(default_factory=list, description="List of improvement suggestions")


class TextToSQLResponse(BaseModel):
    natural_language_query: str = Field(..., description="Original natural language query")
    generated_sql: str = Field(..., description="Generated SQL query")
    query_type: QueryType = Field(..., description="Type of SQL query")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")
    execution_time: float = Field(..., description="Total processing time in seconds")
    model_used: str = Field(..., description="Model used for generation")
    validation_result: Optional[ValidationResult] = Field(None, description="SQL validation results")
    alternative_queries: Optional[List[str]] = Field(
        None, 
        description="Alternative SQL queries if confidence is low"
    )
    explanation: Optional[str] = Field(
        None,
        description="Explanation of how the SQL was generated"
    )
    metadata: Optional[Dict[str, Union[str, int, float]]] = Field(
        None,
        description="Additional metadata about the generation"
    )
    timestamp: datetime = Field(default_factory=datetime.now, description="Generation timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "natural_language_query": "Show me all customers who made purchases in the last month",
                "generated_sql": "SELECT * FROM customers c JOIN orders o ON c.customer_id = o.customer_id WHERE o.order_date >= date('now', '-1 month')",
                "query_type": "SELECT",
                "confidence": 0.85,
                "execution_time": 1.23,
                "model_used": "t5-base",
                "validation_result": {
                    "is_valid": True,
                    "errors": [],
                    "warnings": [],
                    "suggestions": ["Consider adding ORDER BY clause"]
                },
                "alternative_queries": [
                    "SELECT c.*, o.* FROM customers c, orders o WHERE c.customer_id = o.customer_id AND o.order_date >= datetime('now', '-1 month')"
                ],
                "explanation": "Generated SQL joins customers and orders tables, filtering for orders within the last month",
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }


class BatchTextToSQLResponse(BaseModel):
    job_id: str = Field(..., description="Batch job identifier")
    status: str = Field(..., description="Batch processing status")
    total_queries: int = Field(..., description="Total number of queries processed")
    successful_queries: int = Field(..., description="Number of successful queries")
    failed_queries: int = Field(..., description="Number of failed queries")
    results: List[TextToSQLResponse] = Field(..., description="List of individual results")
    processing_time: float = Field(..., description="Total batch processing time")
    created_at: datetime = Field(default_factory=datetime.now, description="Batch creation timestamp")


class QueryValidationRequest(BaseModel):
    sql_query: str = Field(..., description="SQL query to validate")
    database_type: DatabaseType = Field(default=DatabaseType.SQLITE, description="Target database type")
    database_schema: Optional[Dict[str, List[Dict[str, str]]]] = Field(
        None, 
        description="Database schema for validation"
    )
    validation_level: ValidationLevel = Field(default=ValidationLevel.STRICT, description="Validation level")


class QueryValidationResponse(BaseModel):
    sql_query: str = Field(..., description="Original SQL query")
    is_valid: bool = Field(..., description="Whether the SQL is valid")
    database_type: DatabaseType = Field(..., description="Target database type")
    validation_level: ValidationLevel = Field(..., description="Validation level used")
    errors: List[str] = Field(default_factory=list, description="List of validation errors")
    warnings: List[str] = Field(default_factory=list, description="List of validation warnings")
    suggestions: List[str] = Field(default_factory=list, description="List of improvement suggestions")
    corrected_query: Optional[str] = Field(None, description="Corrected SQL query if applicable")
    validation_time: float = Field(..., description="Validation time in seconds")


class ModelInfo(BaseModel):
    model_name: str = Field(..., description="Name of the model")
    model_type: str = Field(..., description="Type of model (T5, BART, etc.)")
    description: str = Field(..., description="Model description")
    max_input_length: int = Field(..., description="Maximum input token length")
    max_output_length: int = Field(..., description="Maximum output token length")
    parameters: str = Field(..., description="Number of parameters")
    supported_languages: List[str] = Field(..., description="Supported languages")
    recommended_use_cases: List[str] = Field(..., description="Recommended use cases")
    
    
class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    models_loaded: List[str] = Field(..., description="List of loaded models")
    available_models: List[str] = Field(..., description="List of available models")
    device: str = Field(..., description="Computing device (cuda/cpu)")
    memory_usage: Optional[Dict[str, float]] = Field(None, description="Memory usage statistics")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for debugging")