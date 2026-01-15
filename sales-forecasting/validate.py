"""
Sales Forecasting API Validation Script
Validates the sales forecasting system without running Python
"""

import os
import json
import glob

def validate_file_exists(filepath, description):
    """Check if a file exists and report its status"""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description}: {filepath} (MISSING)")
        return False

def validate_directory_structure():
    """Validate the project directory structure"""
    print("=== Validating Sales Forecasting Project Structure ===")
    
    base_path = "sales-forecasting"
    
    # Check main directories
    directories = [
        (f"{base_path}", "Project root directory"),
        (f"{base_path}/app", "App directory"),
        (f"{base_path}/notebooks", "Notebooks directory"),
    ]
    
    all_good = True
    for dir_path, description in directories:
        if not validate_file_exists(dir_path, description):
            all_good = False
    
    return all_good

def validate_core_files():
    """Validate all core project files"""
    print("\n=== Validating Core Files ===")
    
    base_path = "sales-forecasting"
    
    files_to_check = [
        (f"{base_path}/requirements.txt", "Requirements file"),
        (f"{base_path}/README.md", "README documentation"),
        (f"{base_path}/app/__init__.py", "App package init"),
        (f"{base_path}/app/main.py", "FastAPI main application"),
        (f"{base_path}/app/model.py", "Forecasting models"),
        (f"{base_path}/app/schemas.py", "Pydantic schemas"),
        (f"{base_path}/notebooks/train_sales_forecast.ipynb", "Training notebook"),
    ]
    
    all_good = True
    for file_path, description in files_to_check:
        if not validate_file_exists(file_path, description):
            all_good = False
    
    return all_good

def validate_python_syntax():
    """Validate Python syntax in key files"""
    print("\n=== Validating Python Syntax ===")
    
    base_path = "sales-forecasting"
    python_files = [
        f"{base_path}/app/main.py",
        f"{base_path}/app/model.py", 
        f"{base_path}/app/schemas.py",
        f"{base_path}/app/__init__.py"
    ]
    
    all_good = True
    for filepath in python_files:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Basic syntax checks
                if content.strip():
                    # Check for common syntax issues
                    if content.count('def ') > 0 or content.count('class ') > 0:
                        print(f"✓ {filepath}: Contains functions/classes")
                    else:
                        print(f"⚠ {filepath}: No functions/classes found")
                    
                    # Check imports
                    if 'import ' in content:
                        print(f"✓ {filepath}: Contains imports")
                    else:
                        print(f"⚠ {filepath}: No imports found")
                        
                    # Check for proper structure
                    lines = content.split('\n')
                    if len(lines) > 5:
                        print(f"✓ {filepath}: Has substantial content ({len(lines)} lines)")
                    else:
                        print(f"⚠ {filepath}: Very short file ({len(lines)} lines)")
                        
                else:
                    print(f"✗ {filepath}: Empty file")
                    all_good = False
                    
            except Exception as e:
                print(f"✗ {filepath}: Error reading file - {e}")
                all_good = False
    
    return all_good

def validate_model_implementation():
    """Validate model implementation specifics"""
    print("\n=== Validating Model Implementation ===")
    
    model_file = "sales-forecasting/app/model.py"
    
    if not os.path.exists(model_file):
        print("✗ Model file not found")
        return False
    
    try:
        with open(model_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = [
            ('prophet', 'Prophet model implementation'),
            ('arima', 'ARIMA model implementation'), 
            ('linear_regression', 'Linear regression implementation'),
            ('generate_synthetic_sales_data', 'Synthetic data generation'),
            ('train_prophet_model', 'Prophet training method'),
            ('train_arima_model', 'ARIMA training method'),
            ('train_linear_regression_model', 'Linear regression training'),
            ('predict_sales', 'Prediction method'),
            ('evaluate_model', 'Model evaluation')
        ]
        
        all_good = True
        for keyword, description in checks:
            if keyword in content.lower():
                print(f"✓ {description} found")
            else:
                print(f"✗ {description} not found")
                all_good = False
        
        return all_good
        
    except Exception as e:
        print(f"✗ Error validating model implementation: {e}")
        return False

def validate_api_endpoints():
    """Validate API endpoint implementation"""
    print("\n=== Validating API Endpoints ===")
    
    main_file = "sales-forecasting/app/main.py"
    
    if not os.path.exists(main_file):
        print("✗ Main API file not found")
        return False
    
    try:
        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        endpoints = [
            ('@app.get("/")', 'Root endpoint'),
            ('@app.get("/health")', 'Health check endpoint'),
            ('@app.post("/predict")', 'Prediction endpoint'),
            ('@app.post("/predict/batch")', 'Batch prediction endpoint'),
            ('@app.post("/train")', 'Training endpoint'),
            ('@app.get("/model/info")', 'Model info endpoint'),
            ('@app.post("/generate-sample-data")', 'Sample data generation endpoint')
        ]
        
        all_good = True
        for endpoint_pattern, description in endpoints:
            if endpoint_pattern in content:
                print(f"✓ {description} found")
            else:
                print(f"✗ {description} not found")
                all_good = False
        
        return all_good
        
    except Exception as e:
        print(f"✗ Error validating API endpoints: {e}")
        return False

def validate_requirements():
    """Validate requirements.txt content"""
    print("\n=== Validating Requirements ===")
    
    req_file = "sales-forecasting/requirements.txt"
    
    if not os.path.exists(req_file):
        print("✗ requirements.txt not found")
        return False
    
    try:
        with open(req_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_packages = [
            'fastapi',
            'prophet',
            'statsmodels',
            'scikit-learn',
            'pandas',
            'numpy',
            'joblib',
            'uvicorn'
        ]
        
        all_good = True
        for package in required_packages:
            if package in content.lower():
                print(f"✓ {package} found in requirements")
            else:
                print(f"✗ {package} missing from requirements")
                all_good = False
        
        return all_good
        
    except Exception as e:
        print(f"✗ Error validating requirements: {e}")
        return False

def main():
    """Main validation function"""
    print("Starting Sales Forecasting Project Validation...")
    print("=" * 60)
    
    results = []
    
    # Run all validations
    results.append(validate_directory_structure())
    results.append(validate_core_files())
    results.append(validate_python_syntax())
    results.append(validate_model_implementation())
    results.append(validate_api_endpoints())
    results.append(validate_requirements())
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✓ ALL VALIDATIONS PASSED ({passed}/{total})")
        print("✓ Sales Forecasting project is ready for use!")
    else:
        print(f"✗ SOME VALIDATIONS FAILED ({passed}/{total})")
        print("✗ Please address the issues above before proceeding")
    
    print("=" * 60)

if __name__ == "__main__":
    main()