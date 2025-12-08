import requests
import json
import numpy as np
import time
from datetime import datetime

def create_test_users():
    """Create test users"""
    users = []
    for i in range(1, 21):  # 20 test users
        user = {
            "user_id": f"user_{i:03d}",
            "age": np.random.randint(18, 65),
            "gender": np.random.choice(["male", "female", "other"]),
            "location": np.random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]),
            "preferences": {
                "genres": np.random.choice(["action", "comedy", "drama", "sci-fi", "romance"], size=3, replace=False).tolist()
            }
        }
        users.append(user)
    return users

def create_test_items():
    """Create test items (movies/products)"""
    items = []
    categories = ["movie", "book", "music", "game"]
    
    for i in range(1, 51):  # 50 test items
        item = {
            "item_id": f"item_{i:03d}",
            "item_name": f"Test Item {i}",
            "category": np.random.choice(categories),
            "tags": np.random.choice(["popular", "new", "classic", "trending"], size=2, replace=False).tolist(),
            "metadata": {
                "release_year": np.random.randint(2010, 2024),
                "rating": round(np.random.uniform(3.0, 5.0), 1)
            }
        }
        items.append(item)
    return items

def create_test_ratings(users, items, num_ratings=200):
    """Create test ratings"""
    ratings = []
    
    for _ in range(num_ratings):
        user = np.random.choice(users)
        item = np.random.choice(items)
        
        # Generate realistic ratings (higher ratings for items that match user preferences)
        base_rating = np.random.normal(3.5, 1.0)
        rating = max(1.0, min(5.0, round(base_rating, 1)))
        
        rating_data = {
            "user_id": user["user_id"],
            "item_id": item["item_id"],
            "rating": rating,
            "timestamp": datetime.now().isoformat()
        }
        ratings.append(rating_data)
    
    return ratings

def test_api_endpoints():
    """Test all API endpoints"""
    base_url = "http://localhost:8000"
    
    print("🚀 Starting Recommendation System API Tests")
    print("=" * 50)
    
    # Test 1: Health Check
    print("\n1. Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Status: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return
    
    # Create test data
    print("\n2. Creating test data...")
    users = create_test_users()
    items = create_test_items()
    ratings = create_test_ratings(users, items, 200)
    
    print(f"   Created {len(users)} users, {len(items)} items, {len(ratings)} ratings")
    
    # Test 2: Add Users
    print("\n3. Testing Add Users...")
    success_count = 0
    for user in users[:5]:  # Test with first 5 users
        try:
            response = requests.post(f"{base_url}/users", json=user)
            if response.status_code == 200:
                success_count += 1
            else:
                print(f"   Warning: Failed to add user {user['user_id']}: {response.status_code}")
        except Exception as e:
            print(f"   Error adding user {user['user_id']}: {str(e)}")
    
    print(f"✅ Added {success_count} users successfully")
    
    # Test 3: Add Items
    print("\n4. Testing Add Items...")
    success_count = 0
    for item in items[:10]:  # Test with first 10 items
        try:
            response = requests.post(f"{base_url}/items", json=item)
            if response.status_code == 200:
                success_count += 1
            else:
                print(f"   Warning: Failed to add item {item['item_id']}: {response.status_code}")
        except Exception as e:
            print(f"   Error adding item {item['item_id']}: {str(e)}")
    
    print(f"✅ Added {success_count} items successfully")
    
    # Test 4: Add Ratings
    print("\n5. Testing Add Ratings...")
    success_count = 0
    for rating in ratings[:50]:  # Test with first 50 ratings
        try:
            response = requests.post(f"{base_url}/ratings", json=rating)
            if response.status_code == 200:
                success_count += 1
            else:
                print(f"   Warning: Failed to add rating: {response.status_code}")
        except Exception as e:
            print(f"   Error adding rating: {str(e)}")
    
    print(f"✅ Added {success_count} ratings successfully")
    
    # Test 5: Batch Ratings
    print("\n6. Testing Batch Ratings...")
    try:
        batch_ratings = ratings[50:100]  # Next 50 ratings
        response = requests.post(f"{base_url}/batch-ratings", json=batch_ratings)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch ratings processed: {result['success_count']} successful, {result['failed_count']} failed")
        else:
            print(f"❌ Batch ratings failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Batch ratings error: {str(e)}")
    
    # Test 6: Get Stats
    print("\n7. Testing System Stats...")
    try:
        response = requests.get(f"{base_url}/stats")
        if response.status_code == 200:
            stats = response.json()
            print("✅ System stats retrieved")
            print(f"   Total Users: {stats['total_users']}")
            print(f"   Total Items: {stats['total_items']}")
            print(f"   Total Ratings: {stats['total_ratings']}")
            print(f"   Sparsity: {stats['sparsity']:.2%}")
        else:
            print(f"❌ Stats retrieval failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Stats error: {str(e)}")
    
    # Test 7: Training
    print("\n8. Testing Model Training...")
    try:
        training_request = {
            "algorithm": "matrix_factorization",
            "hyperparameters": {
                "n_components": 20,
                "max_iter": 100
            }
        }
        
        response = requests.post(f"{base_url}/train", json=training_request)
        if response.status_code == 200:
            result = response.json()
            print("✅ Model training completed")
            print(f"   Algorithm: {result['algorithm']}")
            print(f"   Training Time: {result['training_time_seconds']:.2f} seconds")
            print(f"   Training Samples: {result['training_samples']}")
        else:
            print(f"❌ Training failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Training error: {str(e)}")
    
    # Test 8: Generate Recommendations
    print("\n9. Testing Recommendations...")
    try:
        # Test with first user
        test_user = users[0]['user_id']
        recommendation_request = {
            "user_id": test_user,
            "algorithm": "matrix_factorization",
            "num_recommendations": 5
        }
        
        response = requests.post(f"{base_url}/recommendations", json=recommendation_request)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Recommendations generated for user {test_user}")
            print(f"   Algorithm: {result['algorithm']}")
            print(f"   Total Candidates: {result['total_candidates']}")
            print(f"   Number of Recommendations: {len(result['recommendations'])}")
            
            if result['recommendations']:
                print("   Top Recommendations:")
                for i, rec in enumerate(result['recommendations'][:3], 1):
                    print(f"     {i}. {rec['item_name']} (Score: {rec['predicted_rating']}, Confidence: {rec['confidence']:.2f})")
        else:
            print(f"❌ Recommendations failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Recommendations error: {str(e)}")
    
    # Test 9: Different Algorithms
    print("\n10. Testing Different Algorithms...")
    algorithms = ["collaborative_filtering", "item_based", "hybrid"]
    
    for algorithm in algorithms:
        try:
            test_user = users[1]['user_id']  # Use second user
            recommendation_request = {
                "user_id": test_user,
                "algorithm": algorithm,
                "num_recommendations": 3
            }
            
            response = requests.post(f"{base_url}/recommendations", json=recommendation_request)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {algorithm} recommendations: {len(result['recommendations'])} items")
            else:
                print(f"❌ {algorithm} failed: {response.status_code}")
        except Exception as e:
            print(f"❌ {algorithm} error: {str(e)}")
    
    # Test 10: User Ratings
    print("\n11. Testing User Ratings Retrieval...")
    try:
        test_user = users[0]['user_id']
        response = requests.get(f"{base_url}/users/{test_user}/ratings")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ User ratings retrieved: {result['total_ratings']} ratings for user {test_user}")
        else:
            print(f"❌ User ratings failed: {response.status_code}")
    except Exception as e:
        print(f"❌ User ratings error: {str(e)}")
    
    # Test 11: Item Ratings
    print("\n12. Testing Item Ratings Retrieval...")
    try:
        test_item = items[0]['item_id']
        response = requests.get(f"{base_url}/items/{test_item}/ratings")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Item ratings retrieved: {result['total_ratings']} ratings for item {test_item}")
            if result['total_ratings'] > 0:
                print(f"   Average Rating: {result['average_rating']}")
        else:
            print(f"❌ Item ratings failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Item ratings error: {str(e)}")
    
    # Test 12: Model Evaluation
    print("\n13. Testing Model Evaluation...")
    try:
        # Create some test ratings for evaluation
        test_ratings = []
        for rating in ratings[100:120]:  # Use different ratings for testing
            test_ratings.append({
                "user_id": rating["user_id"],
                "item_id": rating["item_id"],
                "rating": rating["rating"]
            })
        
        evaluation_request = {
            "test_ratings": test_ratings,
            "k": 5
        }
        
        response = requests.post(f"{base_url}/evaluate", json=evaluation_request)
        if response.status_code == 200:
            result = response.json()
            print("✅ Model evaluation completed")
            print(f"   Precision@5: {result['precision_at_k']:.3f}")
            print(f"   Recall@5: {result['recall_at_k']:.3f}")
            print(f"   F1-Score: {result['f1_score']:.3f}")
            print(f"   Test Users: {result['num_test_users']}")
        else:
            print(f"❌ Evaluation failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Evaluation error: {str(e)}")
    
    print("\n" + "=" * 50)
    print("🎉 Recommendation System API Tests Completed!")
    print("\nNext steps:")
    print("- Check the API documentation at: http://localhost:8000/docs")
    print("- Test with your own data")
    print("- Experiment with different algorithms and hyperparameters")

if __name__ == "__main__":
    test_api_endpoints()