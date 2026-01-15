#!/usr/bin/env python3
"""
Validation script for Price Optimization Engine
Checks syntax, imports, and basic functionality
"""

import ast
import sys
import importlib.util
import subprocess
from pathlib import Path

def check_syntax(file_path):
    """Check Python syntax for a file"""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def check_imports(file_path):
    """Check if all imports can be resolved"""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        
        tree = ast.parse(source)
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        missing_imports = []
        for imp in imports:
            try:
                if '.' in imp:
                    # Handle relative imports
                    continue
                importlib.import_module(imp)
            except ImportError:
                missing_imports.append(imp)
        
        if missing_imports:
            return False, f"Missing imports: {missing_imports}"
        return True, None
    except Exception as e:
        return False, f"Import check error: {e}"

def validate_project():
    """Validate the entire Price Optimization Engine project"""
    project_root = Path(__file__).parent
    app_dir = project_root / "app"
    
    print("🔍 Validating Price Optimization Engine Project...")
    print("=" * 60)
    
    # Files to check
    files_to_check = [
        app_dir / "__init__.py",
        app_dir / "schemas.py",
        app_dir / "model.py",
        app_dir / "main.py",
        project_root / "test_api.py"
    ]
    
    all_passed = True
    
    for file_path in files_to_check:
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            all_passed = False
            continue
        
        print(f"\n📄 Checking {file_path.name}...")
        
        # Check syntax
        syntax_ok, syntax_error = check_syntax(file_path)
        if syntax_ok:
            print(f"  ✅ Syntax: OK")
        else:
            print(f"  ❌ Syntax: {syntax_error}")
            all_passed = False
        
        # Check imports
        import_ok, import_error = check_imports(file_path)
        if import_ok:
            print(f"  ✅ Imports: OK")
        else:
            print(f"  ❌ Imports: {import_error}")
            all_passed = False
    
    # Check requirements.txt
    requirements_file = project_root / "requirements.txt"
    if requirements_file.exists():
        print(f"\n📋 Checking requirements.txt...")
        try:
            with open(requirements_file, 'r') as f:
                requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            print(f"  Found {len(requirements)} dependencies:")
            for req in requirements[:5]:  # Show first 5
                print(f"    - {req}")
            if len(requirements) > 5:
                print(f"    ... and {len(requirements) - 5} more")
            print(f"  ✅ Requirements file: OK")
        except Exception as e:
            print(f"  ❌ Requirements file error: {e}")
            all_passed = False
    
    # Test basic functionality
    print(f"\n🧪 Testing basic functionality...")
    try:
        # Test model imports
        sys.path.insert(0, str(project_root))
        
        # Test schema imports
        from app.schemas import PriceRequest, PriceResponse, PriceOptimizationResult
        print(f"  ✅ Schema imports: OK")
        
        # Test model imports
        from app.model import PriceOptimizationModel, QLearningAgent
        print(f"  ✅ Model imports: OK")
        
        # Test basic model initialization
        model = PriceOptimizationModel()
        print(f"  ✅ Model initialization: OK")
        
        # Test Q-learning agent
        agent = QLearningAgent()
        print(f"  ✅ Q-learning agent: OK")
        
    except Exception as e:
        print(f"  ❌ Functionality test error: {e}")
        all_passed = False
    
    # Final result
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All validation checks passed!")
        print("✅ Price Optimization Engine project is ready to use!")
        return True
    else:
        print("❌ Some validation checks failed!")
        print("🔧 Please fix the issues above before proceeding.")
        return False

if __name__ == "__main__":
    success = validate_project()
    sys.exit(0 if success else 1)