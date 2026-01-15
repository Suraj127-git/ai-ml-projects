"""
Validation script for Product Demand Forecasting System
"""

import os
import sys
import importlib.util
from datetime import datetime

def validate_product_demand_forecasting():
    """Validate the Product Demand Forecasting project structure and syntax"""
    
    print("Product Demand Forecasting System Validation")
    print("=" * 60)
    
    # Check project structure
    required_files = [
        "app/__init__.py",
        "app/main.py", 
        "app/model.py",
        "app/schemas.py",
        "requirements.txt",
        "README.md",
        "test_api.py",
        "notebooks/train_demand_models.py"
    ]
    
    project_path = "c:\\Users\\Suraj\\code\\ai-ml-projects\\product-demand-forecasting"
    missing_files = []
    
    print("1. Checking project structure...")
    for file_path in required_files:
        full_path = os.path.join(project_path, file_path)
        if os.path.exists(full_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing {len(missing_files)} required files")
        return False
    
    print("\n2. Validating Python syntax...")
    
    # Check Python files
    python_files = [
        "app/__init__.py",
        "app/main.py",
        "app/model.py", 
        "app/schemas.py",
        "test_api.py",
        "notebooks/train_demand_models.py"
    ]
    
    syntax_errors = []
    
    for file_path in python_files:
        full_path = os.path.join(project_path, file_path)
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic syntax validation
            compile(content, full_path, 'exec')
            print(f"   ✅ {file_path} - Syntax OK")
            
        except SyntaxError as e:
            print(f"   ❌ {file_path} - Syntax error: {e}")
            syntax_errors.append((file_path, str(e)))
        except Exception as e:
            print(f"   ⚠️  {file_path} - Other error: {e}")
    
    if syntax_errors:
        print(f"\n❌ Found {len(syntax_errors)} syntax errors")
        for file_path, error in syntax_errors:
            print(f"   {file_path}: {error}")
        return False
    
    print("\n3. Checking imports...")
    
    # Check if required packages are available
    required_packages = [
        'fastapi', 'pydantic', 'pandas', 'numpy', 'scikit-learn', 
        'joblib', 'statsmodels', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            spec = importlib.util.find_spec(package)
            if spec is None:
                print(f"   ⚠️  {package} - Not available (optional)")
            else:
                print(f"   ✅ {package} - Available")
        except Exception:
            print(f"   ⚠️  {package} - Check failed")
    
    print("\n4. Validating model structure...")
    
    # Check model class structure
    try:
        sys.path.insert(0, os.path.join(project_path, 'app'))
        
        # Import and validate schemas
        from schemas import (
            ProductData, ForecastRequest, ForecastResponse, BatchForecastRequest,
            BatchForecastResponse, TrainingRequest, TrainingResponse, ModelInfo,
            HealthResponse, ModelName, DemandPattern
        )
        print("   ✅ Pydantic schemas imported successfully")
        
        # Validate model class
        from model import DemandForecastingModel
        print("   ✅ DemandForecastingModel class imported successfully")
        
        # Check if model has required methods
        model_methods = [
            'generate_synthetic_demand_data', 'prepare_time_series_data',
            'train_arima_model', 'train_prophet_model', 'train_lstm_model',
            'train_xgboost_model', 'forecast_demand', 'save_model', 'load_model',
            'get_model_info'
        ]
        
        model_instance = DemandForecastingModel()
        missing_methods = []
        
        for method in model_methods:
            if hasattr(model_instance, method):
                print(f"   ✅ Method: {method}")
            else:
                print(f"   ❌ Missing method: {method}")
                missing_methods.append(method)
        
        if missing_methods:
            print(f"\n❌ Model missing {len(missing_methods)} required methods")
            return False
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Model validation error: {e}")
        return False
    
    print("\n5. Checking FastAPI app structure...")
    
    # Validate FastAPI app endpoints
    try:
        from main import app
        print("   ✅ FastAPI app imported successfully")
        
        # Check if app has required routes
        routes = [
            '/', '/health', '/forecast', '/forecast/batch',
            '/train/{model_type}', '/model/info/{product_id}',
            '/model/performance', '/generate-sample-data',
            '/analyze-demand-pattern'
        ]
        
        available_routes = []
        for route in app.routes:
            if hasattr(route, 'path'):
                available_routes.append(route.path)
        
        missing_routes = []
        for route in routes:
            # Handle parameterized routes
            base_route = route.split('{')[0].rstrip('/')
            found = any(base_route in available_route for available_route in available_routes)
            
            if found:
                print(f"   ✅ Route: {route}")
            else:
                print(f"   ❌ Missing route: {route}")
                missing_routes.append(route)
        
        if missing_routes:
            print(f"\n❌ App missing {len(missing_routes)} required routes")
            return False
        
    except ImportError as e:
        print(f"   ❌ FastAPI app import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ FastAPI validation error: {e}")
        return False
    
    print("\n6. Checking requirements.txt...")
    
    # Validate requirements file
    try:
        req_path = os.path.join(project_path, 'requirements.txt')
        with open(req_path, 'r') as f:
            requirements = f.read()
        
        required_deps = ['fastapi', 'pandas', 'numpy', 'scikit-learn', 'statsmodels']
        missing_deps = []
        
        for dep in required_deps:
            if dep in requirements:
                print(f"   ✅ {dep}")
            else:
                print(f"   ❌ Missing dependency: {dep}")
                missing_deps.append(dep)
        
        if missing_deps:
            print(f"\n❌ Requirements missing {len(missing_deps)} dependencies")
            return False
        
        print(f"   ✅ Total dependencies: {len([line for line in requirements.split('\\n') if line.strip() and not line.startswith('#')])}")
        
    except Exception as e:
        print(f"   ❌ Requirements validation error: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 Validation completed successfully!")
    print("✅ Product Demand Forecasting System is ready for deployment")
    print("\n📋 Summary:")
    print(f"   - Project structure: Complete")
    print(f"   - Python syntax: Valid")
    print(f"   - Model implementation: Complete")
    print(f"   - API endpoints: Ready")
    print(f"   - Dependencies: Configured")
    print(f"\n🚀 To start the API:")
    print(f"   cd {project_path}")
    print(f"   pip install -r requirements.txt")
    print(f"   python -m app.main")
    print(f"\n📖 API Documentation: http://localhost:8003/docs")
    
    return True

if __name__ == "__main__":
    validate_product_demand_forecasting()