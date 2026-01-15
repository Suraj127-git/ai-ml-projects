#!/usr/bin/env python3
"""
Validation script for Supply Chain Optimization API
Tests syntax, imports, and basic functionality
"""

import sys
import os
import importlib.util
import traceback
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    required_modules = [
        'fastapi',
        'pydantic',
        'numpy',
        'pandas',
        'sklearn',
        'joblib'
    ]
    
    optional_modules = [
        'pulp',
        'networkx'
    ]
    
    # Test required modules
    for module in required_modules:
        try:
            __import__(module)
            print(f"   ✓ {module}")
        except ImportError as e:
            print(f"   ✗ {module}: {e}")
            return False
    
    # Test optional modules
    for module in optional_modules:
        try:
            __import__(module)
            print(f"   ✓ {module} (optional)")
        except ImportError:
            print(f"   ~ {module} (optional, not available)")
    
    return True

def test_schemas():
    """Test schema definitions"""
    print("\nTesting schemas...")
    
    try:
        # Import schemas module
        spec = importlib.util.spec_from_file_location(
            "schemas", 
            Path(__file__).parent / "app" / "schemas.py"
        )
        schemas = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(schemas)
        
        # Test enum creation
        objective = schemas.OptimizationObjective.MINIMIZE_COST
        print(f"   ✓ OptimizationObjective: {objective}")
        
        # Test model creation
        node = schemas.SupplyChainNode(
            node_id="test_node",
            name="Test Node",
            node_type=schemas.NodeType.SUPPLIER,
            location={"lat": 40.0, "lng": -74.0},
            capacity=100,
            setup_cost=1000
        )
        print(f"   ✓ SupplyChainNode: {node.node_id}")
        
        # Test optimization request
        request = schemas.OptimizationRequest(
            nodes=[node],
            edges=[],
            demands=[],
            objective=schemas.OptimizationObjective.MINIMIZE_COST
        )
        print(f"   ✓ OptimizationRequest: {request.objective}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Schema test failed: {e}")
        traceback.print_exc()
        return False

def test_model():
    """Test model functionality"""
    print("\nTesting model...")
    
    try:
        # Import model module
        spec = importlib.util.spec_from_file_location(
            "model", 
            Path(__file__).parent / "app" / "model.py"
        )
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        
        # Create optimizer instance
        optimizer = model_module.SupplyChainOptimizer()
        print("   ✓ SupplyChainOptimizer created")
        
        # Test network flow optimization with minimal data
        from app.schemas import (
            SupplyChainNode, SupplyChainEdge, DemandForecast,
            OptimizationRequest, OptimizationObjective, NodeType
        )
        
        # Create minimal test data
        node = SupplyChainNode(
            node_id="test_supplier",
            name="Test Supplier",
            node_type=NodeType.SUPPLIER,
            location={"lat": 40.0, "lng": -74.0},
            capacity=100,
            setup_cost=1000
        )
        
        edge = SupplyChainEdge(
            from_node="test_supplier",
            to_node="test_retailer",
            capacity=50,
            transportation_cost=1.0,
            lead_time=1,
            distance=10,
            carbon_factor=0.1,
            reliability=0.95
        )
        
        demand = DemandForecast(
            node_id="test_retailer",
            product_id="test_product",
            quantity=30,
            mean_demand=30,
            std_demand=5,
            probability_exceeding=0.1
        )
        
        request = OptimizationRequest(
            nodes=[node],
            edges=[edge],
            demands=[demand],
            objective=OptimizationObjective.MINIMIZE_COST
        )
        
        # Test optimization
        result = optimizer.solve_network_flow_optimization(request)
        print(f"   ✓ Network optimization: {result.optimization_status}")
        print(f"   - Total cost: ${result.total_cost:.2f}")
        print(f"   - Execution time: {result.execution_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Model test failed: {e}")
        traceback.print_exc()
        return False

def test_api_structure():
    """Test API structure"""
    print("\nTesting API structure...")
    
    try:
        # Import main module
        spec = importlib.util.spec_from_file_location(
            "main", 
            Path(__file__).parent / "app" / "main.py"
        )
        main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_module)
        
        # Check if FastAPI app exists
        if hasattr(main_module, 'app'):
            print(f"   ✓ FastAPI app found: {main_module.app.title}")
        else:
            print("   ✗ FastAPI app not found")
            return False
        
        # Check if optimizer exists
        if hasattr(main_module, 'optimizer'):
            print("   ✓ SupplyChainOptimizer instance found")
        else:
            print("   ✗ SupplyChainOptimizer instance not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ✗ API structure test failed: {e}")
        traceback.print_exc()
        return False

def test_file_structure():
    """Test that all required files exist"""
    print("\nTesting file structure...")
    
    required_files = [
        "app/__init__.py",
        "app/schemas.py",
        "app/model.py",
        "app/main.py",
        "requirements.txt",
        "test_api.py",
        "validate.py"
    ]
    
    base_path = Path(__file__).parent
    all_exist = True
    
    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"   ✓ {file_path}")
        else:
            print(f"   ✗ {file_path} (missing)")
            all_exist = False
    
    return all_exist

def main():
    """Run all validation tests"""
    print("Supply Chain Optimization API Validation")
    print("=" * 45)
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    # Run tests
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Schemas", test_schemas),
        ("Model", test_model),
        ("API Structure", test_api_structure)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\nUnexpected error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 45)
    print("Validation Summary:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"   {status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All validation tests passed!")
        return 0
    else:
        print("❌ Some validation tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())