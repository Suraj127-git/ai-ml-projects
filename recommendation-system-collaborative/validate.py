#!/usr/bin/env python3
"""
Validation script for the Recommendation System (Collaborative) project.
This script checks imports, validates schemas, and tests basic functionality.
"""

import sys
import traceback
from datetime import datetime

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        from sklearn.decomposition import NMF
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.preprocessing import StandardScaler
        print("✅ scikit-learn components imported successfully")
    except ImportError as e:
        print(f"❌ scikit-learn import failed: {e}")
        return False
    
    try:
        from fastapi import FastAPI, HTTPException
        print("✅ fastapi imported successfully")
    except ImportError as e:
        print(f"❌ fastapi import failed: {e}")
        return False
    
    try:
        from pydantic import BaseModel, Field
        print("✅ pydantic imported successfully")
    except ImportError as e:
        print(f"❌ pydantic import failed: {e}")
        return False
    
    return True

def test_schemas():
    """Test schema definitions"""
    print("\n📋 Testing schemas...")
    
    try:
        from app.schemas import (
            User, Item, Rating, RecommendationRequest, TrainingRequest,
            RecommendationResponse, TrainingResponse, EvaluationResponse,
            RecommendationAlgorithm
        )
        print("✅ All schemas imported successfully")
        
        # Test User schema
        user = User(
            user_id="test_user_001",
            age=25,
            gender="male",
            location="New York",
            preferences={"genres": ["action", "comedy"]}
        )
        print(f"✅ User schema works: {user.user_id}")
        
        # Test Item schema
        item = Item(
            item_id="test_item_001",
            item_name="Test Movie",
            category="movie",
            tags=["action", "popular"],
            metadata={"year": 2023}
        )
        print(f"✅ Item schema works: {item.item_id}")
        
        # Test Rating schema
        rating = Rating(
            user_id="test_user_001",
            item_id="test_item_001",
            rating=4.5
        )
        print(f"✅ Rating schema works: {rating.rating}")
        
        # Test RecommendationRequest schema
        rec_request = RecommendationRequest(
            user_id="test_user_001",
            algorithm=RecommendationAlgorithm.MATRIX_FACTORIZATION,
            num_recommendations=10
        )
        print(f"✅ RecommendationRequest schema works: {rec_request.algorithm}")
        
        return True
        
    except Exception as e:
        print(f"❌ Schema test failed: {e}")
        traceback.print_exc()
        return False

def test_model():
    """Test the recommendation engine model"""
    print("\n🧠 Testing recommendation engine...")
    
    try:
        from app.model import RecommendationEngine
        
        # Create engine instance
        engine = RecommendationEngine()
        print("✅ RecommendationEngine created successfully")
        
        # Test adding users
        from app.schemas import User
        user = User(user_id="user_001", age=25, gender="male", location="NYC")
        success = engine.add_user(user)
        if success:
            print("✅ User added successfully")
        else:
            print("❌ Failed to add user")
            return False
        
        # Test adding items
        from app.schemas import Item
        item = Item(item_id="item_001", item_name="Test Item", category="movie")
        success = engine.add_item(item)
        if success:
            print("✅ Item added successfully")
        else:
            print("❌ Failed to add item")
            return False
        
        # Test adding ratings
        from app.schemas import Rating
        rating = Rating(user_id="user_001", item_id="item_001", rating=4.5)
        success = engine.add_rating(rating)
        if success:
            print("✅ Rating added successfully")
        else:
            print("❌ Failed to add rating")
            return False
        
        # Test training (with minimal data)
        from app.schemas import TrainingRequest
        
        # Add more ratings for training
        for i in range(2, 11):
            user = User(user_id=f"user_{i:03d}", age=20+i, gender="female", location="NYC")
            engine.add_user(user)
            
            item = Item(item_id=f"item_{i:03d}", item_name=f"Test Item {i}", category="movie")
            engine.add_item(item)
            
            rating = Rating(user_id=f"user_{i:03d}", item_id=f"item_{i:03d}", rating=np.random.uniform(1, 5))
            engine.add_rating(rating)
        
        # Add some cross-ratings
        for i in range(1, 6):
            for j in range(1, 6):
                if i != j:
                    rating = Rating(
                        user_id=f"user_{i:03d}", 
                        item_id=f"item_{j:03d}", 
                        rating=np.random.uniform(1, 5)
                    )
                    engine.add_rating(rating)
        
        training_request = TrainingRequest(
            algorithm="matrix_factorization",
            hyperparameters={"n_components": 5, "max_iter": 50}
        )
        
        response = engine.train(training_request)
        if response.status == "success":
            print("✅ Model trained successfully")
            print(f"   Training time: {response.training_time_seconds:.2f}s")
            print(f"   Training samples: {response.training_samples}")
        else:
            print(f"❌ Training failed: {response.error_message}")
            return False
        
        # Test recommendations
        from app.schemas import RecommendationRequest
        rec_request = RecommendationRequest(
            user_id="user_001",
            algorithm="matrix_factorization",
            num_recommendations=5
        )
        
        response = engine.generate_recommendations(rec_request)
        if not response.error_message:
            print("✅ Recommendations generated successfully")
            print(f"   Generated {len(response.recommendations)} recommendations")
        else:
            print(f"❌ Recommendation failed: {response.error_message}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        traceback.print_exc()
        return False

def test_fastapi_app():
    """Test FastAPI app creation"""
    print("\n⚡ Testing FastAPI app...")
    
    try:
        from app.main import app
        print("✅ FastAPI app imported successfully")
        
        # Test app configuration
        if hasattr(app, 'title'):
            print(f"✅ App title: {app.title}")
        
        # Test that routes exist
        routes = [route.path for route in app.routes]
        expected_routes = ["/", "/health", "/users", "/items", "/ratings", "/train", "/recommendations"]
        
        for route in expected_routes:
            if route in routes:
                print(f"✅ Route {route} exists")
            else:
                print(f"❌ Route {route} missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ FastAPI app test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all validation tests"""
    print("🚀 Starting Recommendation System Validation")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Imports", test_imports),
        ("Schemas", test_schemas),
        ("Model", test_model),
        ("FastAPI App", test_fastapi_app)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:15} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The recommendation system is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())