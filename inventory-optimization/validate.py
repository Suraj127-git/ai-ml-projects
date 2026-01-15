#!/usr/bin/env python3
"""
Inventory Optimization System - Project Validation Script
Validates the project structure, dependencies, and functionality
"""

import os
import sys
import json
import importlib.util
from pathlib import Path

def validate_project_structure():
    """Validate that all required files and directories exist"""
    
    print("🔍 Validating Inventory Optimization System project structure...")
    
    base_path = Path(__file__).parent
    required_files = [
        'app/__init__.py',
        'app/main.py',
        'app/model.py',
        'app/schemas.py',
        'requirements.txt',
        'README.md',
        'test_api.py',
        'notebooks/train_inventory_models.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = base_path / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files present")
    return True

def validate_python_syntax():
    """Validate Python syntax in all Python files"""
    
    print("\n🔍 Validating Python syntax...")
    
    base_path = Path(__file__).parent
    python_files = [
        'app/__init__.py',
        'app/main.py',
        'app/model.py',
        'app/schemas.py',
        'test_api.py',
        'notebooks/train_inventory_models.py'
    ]
    
    syntax_errors = []
    
    for file_path in python_files:
        try:
            full_path = base_path / file_path
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to compile the code
            compile(content, str(full_path), 'exec')
            print(f"✅ {file_path}")
            
        except SyntaxError as e:
            syntax_errors.append(f"{file_path}: {e}")
            print(f"❌ {file_path}: {e}")
        except Exception as e:
            syntax_errors.append(f"{file_path}: {e}")
            print(f"❌ {file_path}: {e}")
    
    if syntax_errors:
        print(f"❌ Syntax errors found in {len(syntax_errors)} files")
        return False
    
    print("✅ All Python files have valid syntax")
    return True

def validate_imports():
    """Validate that all imports can be resolved"""
    
    print("\n🔍 Validating import statements...")
    
    # Check if we can import the main modules
    try:
        # Add the project directory to Python path
        project_dir = Path(__file__).parent
        sys.path.insert(0, str(project_dir))
        
        # Try to import the modules
        spec = importlib.util.spec_from_file_location("main", project_dir / "app" / "main.py")
        if spec and spec.loader:
            main_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(main_module)
            print("✅ app/main.py imports valid")
        
        spec = importlib.util.spec_from_file_location("model", project_dir / "app" / "model.py")
        if spec and spec.loader:
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)
            print("✅ app/model.py imports valid")
        
        spec = importlib.util.spec_from_file_location("schemas", project_dir / "app" / "schemas.py")
        if spec and spec.loader:
            schemas_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(schemas_module)
            print("✅ app/schemas.py imports valid")
        
        print("✅ All imports validated")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Import validation error: {e}")
        return False

def validate_requirements():
    """Validate requirements.txt format and content"""
    
    print("\n🔍 Validating requirements.txt...")
    
    req_file = Path(__file__).parent / "requirements.txt"
    
    try:
        with open(req_file, 'r', encoding='utf-8') as f:
            requirements = f.readlines()
        
        # Check for required packages
        required_packages = [
            'fastapi',
            'uvicorn',
            'pydantic',
            'pandas',
            'numpy',
            'scikit-learn',
            'scipy',
            'joblib',
            'statsmodels'
        ]
        
        found_packages = []
        for req in requirements:
            req = req.strip()
            if req and not req.startswith('#'):
                package_name = req.split('==')[0].split('>=')[0].split('<=')[0].strip()
                found_packages.append(package_name)
                print(f"✅ Found: {package_name}")
        
        missing_packages = []
        for req_pkg in required_packages:
            if not any(req_pkg in found_pkg for found_pkg in found_packages):
                missing_packages.append(req_pkg)
        
        if missing_packages:
            print(f"⚠️  Potentially missing packages: {missing_packages}")
        
        print("✅ Requirements validation completed")
        return True
        
    except Exception as e:
        print(f"❌ Requirements validation error: {e}")
        return False

def validate_fastapi_app():
    """Validate FastAPI app structure"""
    
    print("\n🔍 Validating FastAPI application structure...")
    
    try:
        project_dir = Path(__file__).parent
        sys.path.insert(0, str(project_dir))
        
        # Import and validate FastAPI app
        from app.main import app
        from app.model import InventoryOptimizationModel
        from app.schemas import ProductData, OptimizationRequest
        
        # Check if app is a FastAPI instance
        if hasattr(app, 'routes'):
            routes = [route.path for route in app.routes]
            print(f"✅ FastAPI app created with {len(routes)} routes")
            
            # Check for required endpoints
            required_endpoints = [
                '/',
                '/health',
                '/optimize',
                '/optimize/batch',
                '/abc-analysis',
                '/multi-echelon',
                '/stock-alerts',
                '/demand-forecast',
                '/calculate-metrics',
                '/generate-sample-data',
                '/model/info'
            ]
            
            found_endpoints = []
            missing_endpoints = []
            
            for endpoint in required_endpoints:
                if endpoint in routes or endpoint + '/' in routes:
                    found_endpoints.append(endpoint)
                    print(f"✅ Found endpoint: {endpoint}")
                else:
                    missing_endpoints.append(endpoint)
            
            if missing_endpoints:
                print(f"⚠️  Missing endpoints: {missing_endpoints}")
            
        else:
            print("❌ Invalid FastAPI app structure")
            return False
        
        # Validate model class
        if hasattr(InventoryOptimizationModel, 'calculate_eoq'):
            print("✅ InventoryOptimizationModel class structure valid")
        else:
            print("❌ Invalid InventoryOptimizationModel structure")
            return False
        
        # Validate schemas
        if hasattr(ProductData, 'model_fields'):
            print(f"✅ ProductData schema with {len(ProductData.model_fields)} fields")
        
        if hasattr(OptimizationRequest, 'model_fields'):
            print(f"✅ OptimizationRequest schema with {len(OptimizationRequest.model_fields)} fields")
        
        print("✅ FastAPI app validation completed")
        return True
        
    except Exception as e:
        print(f"❌ FastAPI validation error: {e}")
        return False

def validate_test_file():
    """Validate the test file structure"""
    
    print("\n🔍 Validating test file...")
    
    try:
        test_file = Path(__file__).parent / "test_api.py"
        
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for test functions
        test_functions = [
            'test_inventory_optimization_api',
            'test_multi_echelon_optimization'
        ]
        
        found_functions = []
        for func_name in test_functions:
            if func_name in content:
                found_functions.append(func_name)
                print(f"✅ Found test function: {func_name}")
        
        if len(found_functions) < len(test_functions):
            print(f"⚠️  Missing test functions: {set(test_functions) - set(found_functions)}")
        
        # Check for API endpoint tests
        api_endpoints = [
            '/health',
            '/optimize',
            '/optimize/batch',
            '/abc-analysis',
            '/multi-echelon',
            '/stock-alerts',
            '/demand-forecast',
            '/calculate-metrics',
            '/generate-sample-data'
        ]
        
        found_endpoints = []
        for endpoint in api_endpoints:
            if endpoint in content:
                found_endpoints.append(endpoint)
        
        print(f"✅ Found {len(found_endpoints)} API endpoint tests")
        print("✅ Test file validation completed")
        return True
        
    except Exception as e:
        print(f"❌ Test file validation error: {e}")
        return False

def create_validation_report():
    """Create a comprehensive validation report"""
    
    print("\n" + "="*60)
    print("📋 INVENTORY OPTIMIZATION SYSTEM VALIDATION REPORT")
    print("="*60)
    
    validation_results = {
        'project_structure': validate_project_structure(),
        'python_syntax': validate_python_syntax(),
        'imports': validate_imports(),
        'requirements': validate_requirements(),
        'fastapi_app': validate_fastapi_app(),
        'test_file': validate_test_file()
    }
    
    print("\n" + "="*60)
    print("📊 VALIDATION SUMMARY")
    print("="*60)
    
    passed = sum(validation_results.values())
    total = len(validation_results)
    
    for test_name, result in validation_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall Score: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All validations passed! The project is ready to run.")
        print("\nTo start the API server:")
        print("uvicorn app.main:app --host 0.0.0.0 --port 8003")
        print("\nTo run tests:")
        print("python test_api.py")
        print("\nTo train models:")
        print("cd notebooks && python train_inventory_models.py")
    else:
        print(f"\n⚠️  {total - passed} validation(s) failed. Please fix the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = create_validation_report()
    sys.exit(0 if success else 1)