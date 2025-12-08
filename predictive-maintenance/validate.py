"""
Validation script for Predictive Maintenance System
Checks project structure, dependencies, and core functionality
"""

import os
import sys
import importlib.util
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description}: {filepath} (NOT FOUND)")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists"""
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        print(f"✓ {description}: {dirpath}")
        return True
    else:
        print(f"✗ {description}: {dirpath} (NOT FOUND)")
        return False

def check_python_syntax(filepath):
    """Check Python file syntax"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Try to compile the code
        compile(content, filepath, 'exec')
        print(f"✓ Syntax check passed: {filepath}")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error in {filepath}: {e}")
        return False
    except Exception as e:
        print(f"✗ Error checking {filepath}: {e}")
        return False

def check_import(module_name, description):
    """Check if a module can be imported"""
    try:
        spec = importlib.util.spec_from_file_location(
            module_name, 
            f"app/{module_name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"✓ Import check passed: {description}")
        return True
    except Exception as e:
        print(f"✗ Import error for {description}: {e}")
        return False

def validate_project_structure():
    """Validate the project structure"""
    print("=" * 60)
    print("PREDICTIVE MAINTENANCE SYSTEM VALIDATION")
    print("=" * 60)
    
    validation_passed = True
    
    # Check project directory
    project_dir = Path(".")
    print(f"\nProject Directory: {project_dir.absolute()}")
    
    # Check main directories
    directories_to_check = [
        ("app", "Application package"),
    ]
    
    for dir_name, description in directories_to_check:
        if not check_directory_exists(dir_name, description):
            validation_passed = False
    
    # Check core files
    files_to_check = [
        ("requirements.txt", "Dependencies file"),
        ("README.md", "Documentation"),
        ("test_api.py", "API tests"),
        ("app/__init__.py", "Package initialization"),
        ("app/main.py", "FastAPI application"),
        ("app/model.py", "Core ML models"),
        ("app/schemas.py", "Pydantic schemas"),
    ]
    
    for file_name, description in files_to_check:
        if not check_file_exists(file_name, description):
            validation_passed = False
    
    return validation_passed

def validate_python_syntax():
    """Validate Python syntax in all files"""
    print("\n" + "=" * 40)
    print("PYTHON SYNTAX VALIDATION")
    print("=" * 40)
    
    validation_passed = True
    python_files = [
        "app/__init__.py",
        "app/main.py", 
        "app/model.py",
        "app/schemas.py",
        "test_api.py"
    ]
    
    for file_path in python_files:
        if os.path.exists(file_path):
            if not check_python_syntax(file_path):
                validation_passed = False
    
    return validation_passed

def validate_imports():
    """Validate that modules can be imported"""
    print("\n" + "=" * 40)
    print("IMPORT VALIDATION")
    print("=" * 40)
    
    validation_passed = True
    
    # Check core module imports
    modules_to_check = [
        ("__init__", "Package initialization"),
        ("schemas", "Pydantic schemas"),
        ("model", "ML models"),
    ]
    
    for module_name, description in modules_to_check:
        if not check_import(module_name, description):
            validation_passed = False
    
    return validation_passed

def validate_model_implementation():
    """Validate the model implementation"""
    print("\n" + "=" * 40)
    print("MODEL IMPLEMENTATION VALIDATION")
    print("=" * 40)
    
    validation_passed = True
    
    try:
        # Import the model module
        spec = importlib.util.spec_from_file_location("model", "app/model.py")
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        
        # Check if PredictiveMaintenanceModel class exists
        if hasattr(model_module, 'PredictiveMaintenanceModel'):
            print("✓ PredictiveMaintenanceModel class found")
            
            # Check key methods
            model_class = model_module.PredictiveMaintenanceModel
            required_methods = [
                'generate_synthetic_maintenance_data',
                'train_random_forest_model',
                'train_gradient_boosting_model', 
                'train_logistic_regression_model',
                'predict_failure_probability',
                'generate_maintenance_schedule',
                'get_recommended_actions',
                'save_model'
            ]
            
            for method in required_methods:
                if hasattr(model_class, method):
                    print(f"✓ Method found: {method}")
                else:
                    print(f"✗ Method missing: {method}")
                    validation_passed = False
        else:
            print("✗ PredictiveMaintenanceModel class not found")
            validation_passed = False
            
    except Exception as e:
        print(f"✗ Error validating model implementation: {e}")
        validation_passed = False
    
    return validation_passed

def validate_api_endpoints():
    """Validate API endpoint definitions"""
    print("\n" + "=" * 40)
    print("API ENDPOINT VALIDATION")
    print("=" * 40)
    
    validation_passed = True
    
    try:
        # Read the main.py file
        with open("app/main.py", 'r') as f:
            content = f.read()
        
        # Check for key endpoints
        required_endpoints = [
            ('@app.get("/")', 'Root endpoint'),
            ('@app.post("/predict")', 'Predict endpoint'),
            ('@app.post("/predict/{model_type}")', 'Predict with model endpoint'),
            ('@app.get("/models")', 'Get models endpoint'),
            ('@app.get("/model-info/{model_type}")', 'Model info endpoint'),
            ('@app.post("/generate-sample-data")', 'Sample data endpoint'),
            ('@app.get("/health")', 'Health check endpoint'),
            ('@app.post("/train/{model_type}")', 'Training endpoint'),
        ]
        
        for endpoint_pattern, description in required_endpoints:
            if endpoint_pattern in content:
                print(f"✓ Endpoint found: {description}")
            else:
                print(f"✗ Endpoint missing: {description}")
                validation_passed = False
        
        # Check for CORS middleware
        if 'CORSMiddleware' in content:
            print("✓ CORS middleware configured")
        else:
            print("✗ CORS middleware not found")
            validation_passed = False
        
        # Check for model loading
        if 'load_models()' in content:
            print("✓ Model loading implemented")
        else:
            print("✗ Model loading not found")
            validation_passed = False
            
    except Exception as e:
        print(f"✗ Error validating API endpoints: {e}")
        validation_passed = False
    
    return validation_passed

def validate_schemas():
    """Validate Pydantic schemas"""
    print("\n" + "=" * 40)
    print("SCHEMA VALIDATION")
    print("=" * 40)
    
    validation_passed = True
    
    try:
        # Read the schemas.py file
        with open("app/schemas.py", 'r') as f:
            content = f.read()
        
        # Check for key schemas
        required_schemas = [
            ('class EquipmentData', 'Equipment data schema'),
            ('class PredictionRequest', 'Prediction request schema'),
            ('class PredictionResponse', 'Prediction response schema'),
            ('class MaintenanceSchedule', 'Maintenance schedule schema'),
            ('class ModelInfo', 'Model info schema'),
            ('class HealthResponse', 'Health response schema'),
        ]
        
        for schema_pattern, description in required_schemas:
            if schema_pattern in content:
                print(f"✓ Schema found: {description}")
            else:
                print(f"✗ Schema missing: {description}")
                validation_passed = False
        
        # Check for enums
        required_enums = [
            ('class EquipmentType', 'Equipment type enum'),
            ('class RiskLevel', 'Risk level enum'),
            ('class MaintenanceType', 'Maintenance type enum'),
            ('class EnvironmentalCondition', 'Environmental condition enum'),
        ]
        
        for enum_pattern, description in required_enums:
            if enum_pattern in content:
                print(f"✓ Enum found: {description}")
            else:
                print(f"✗ Enum missing: {description}")
                validation_passed = False
            
    except Exception as e:
        print(f"✗ Error validating schemas: {e}")
        validation_passed = False
    
    return validation_passed

def main():
    """Main validation function"""
    print("Starting Predictive Maintenance System validation...")
    
    # Run all validations
    validations = [
        ("Project Structure", validate_project_structure),
        ("Python Syntax", validate_python_syntax),
        ("Imports", validate_imports),
        ("Model Implementation", validate_model_implementation),
        ("API Endpoints", validate_api_endpoints),
        ("Schemas", validate_schemas),
    ]
    
    all_passed = True
    
    for validation_name, validation_func in validations:
        if not validation_func():
            all_passed = False
    
    # Final summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    if all_passed:
        print("✓ ALL VALIDATIONS PASSED")
        print("The Predictive Maintenance System is ready for use!")
        return 0
    else:
        print("✗ SOME VALIDATIONS FAILED")
        print("Please fix the issues above before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())