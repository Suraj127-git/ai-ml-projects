#!/usr/bin/env python3
"""
Validation script for Image Classification for Products API
Checks syntax, imports, and basic functionality
"""

import ast
import sys
import os

def validate_python_syntax(file_path):
    """Validate Python syntax for a file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Parse the AST to check for syntax errors
        ast.parse(content)
        print(f"✓ {file_path}: Syntax OK")
        return True
        
    except SyntaxError as e:
        print(f"✗ {file_path}: Syntax Error - {e}")
        return False
    except Exception as e:
        print(f"✗ {file_path}: Error - {e}")
        return False

def validate_imports():
    """Validate that required imports work"""
    required_imports = [
        ("fastapi", "FastAPI"),
        ("pydantic", "BaseModel"),
        ("torch", None),
        ("torchvision", "models"),
        ("PIL", "Image"),
        ("numpy", None),
        ("sklearn.metrics", "accuracy_score")
    ]
    
    all_imports_ok = True
    
    for module, specific_import in required_imports:
        try:
            if specific_import:
                exec(f"from {module} import {specific_import}")
            else:
                exec(f"import {module}")
            print(f"✓ Import {module}{'.' + specific_import if specific_import else ''}: OK")
        except ImportError as e:
            print(f"✗ Import {module}{'.' + specific_import if specific_import else ''}: Failed - {e}")
            all_imports_ok = False
    
    return all_imports_ok

def validate_schemas():
    """Validate Pydantic schemas"""
    try:
        from app.schemas import (
            ClassificationRequest, ClassificationResponse, ProductCategory, ModelType
        )
        
        # Test schema instantiation
        test_request = ClassificationRequest(
            image_data="test_base64_data",
            model_type=ModelType.EFFICIENTNET_B0,
            top_k=5,
            confidence_threshold=0.1,
            image_size="small"
        )
        
        print("✓ Pydantic schemas: OK")
        return True
        
    except Exception as e:
        print(f"✗ Pydantic schemas: Error - {e}")
        return False

def validate_model_architecture():
    """Validate model architecture definitions"""
    try:
        from app.model import ImageClassificationModel, SimpleCNN
        
        # Test model initialization
        model = ImageClassificationModel()
        print("✓ Model architecture: OK")
        return True
        
    except Exception as e:
        print(f"✗ Model architecture: Error - {e}")
        return False

def validate_file_structure():
    """Validate project file structure"""
    required_files = [
        "app/__init__.py",
        "app/main.py",
        "app/model.py",
        "app/schemas.py",
        "requirements.txt",
        "test_api.py",
        "README.md"
    ]
    
    all_files_exist = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}: Exists")
        else:
            print(f"✗ {file_path}: Missing")
            all_files_exist = False
    
    return all_files_exist

def main():
    """Main validation function"""
    print("Image Classification for Products - Validation Script")
    print("=" * 60)
    
    # Change to project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    print(f"Validating project in: {project_dir}")
    print()
    
    # File structure validation
    print("1. Validating file structure...")
    structure_ok = validate_file_structure()
    print()
    
    # Python syntax validation
    print("2. Validating Python syntax...")
    python_files = [
        "app/__init__.py",
        "app/main.py",
        "app/model.py",
        "app/schemas.py",
        "test_api.py"
    ]
    
    syntax_ok = True
    for file_path in python_files:
        if os.path.exists(file_path):
            if not validate_python_syntax(file_path):
                syntax_ok = False
    print()
    
    # Import validation
    print("3. Validating imports...")
    imports_ok = validate_imports()
    print()
    
    # Schema validation
    print("4. Validating Pydantic schemas...")
    schemas_ok = validate_schemas()
    print()
    
    # Model architecture validation
    print("5. Validating model architecture...")
    model_ok = validate_model_architecture()
    print()
    
    # Summary
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    results = [
        ("File Structure", structure_ok),
        ("Python Syntax", syntax_ok),
        ("Imports", imports_ok),
        ("Pydantic Schemas", schemas_ok),
        ("Model Architecture", model_ok)
    ]
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(1 for _, result in results if result)
    
    print(f"\nOverall: {passed_tests}/{total_tests} validation checks passed")
    
    if passed_tests == total_tests:
        print("🎉 All validation checks passed!")
        return 0
    else:
        print(f"⚠️  {total_tests - passed_tests} validation checks failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())