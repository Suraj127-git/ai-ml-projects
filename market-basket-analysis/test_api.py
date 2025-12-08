import requests
import json
import time

def create_test_transactions():
    """Create test transaction data for market basket analysis"""
    transactions = [
        {
            "transaction_id": "T001",
            "timestamp": "2024-01-01T10:00:00Z",
            "items": [
                {"item_id": "bread", "item_name": "White Bread", "category": "bakery", "price": 2.99, "quantity": 1},
                {"item_id": "milk", "item_name": "Whole Milk", "category": "dairy", "price": 3.49, "quantity": 1},
                {"item_id": "eggs", "item_name": "Large Eggs", "category": "dairy", "price": 4.99, "quantity": 1}
            ]
        },
        {
            "transaction_id": "T002",
            "timestamp": "2024-01-01T11:30:00Z",
            "items": [
                {"item_id": "bread", "item_name": "White Bread", "category": "bakery", "price": 2.99, "quantity": 1},
                {"item_id": "butter", "item_name": "Salted Butter", "category": "dairy", "price": 4.49, "quantity": 1}
            ]
        },
        {
            "transaction_id": "T003",
            "timestamp": "2024-01-01T14:15:00Z",
            "items": [
                {"item_id": "milk", "item_name": "Whole Milk", "category": "dairy", "price": 3.49, "quantity": 2},
                {"item_id": "cereal", "item_name": "Corn Flakes", "category": "breakfast", "price": 3.99, "quantity": 1},
                {"item_id": "bananas", "item_name": "Bananas", "category": "produce", "price": 1.99, "quantity": 6}
            ]
        },
        {
            "transaction_id": "T004",
            "timestamp": "2024-01-01T16:45:00Z",
            "items": [
                {"item_id": "bread", "item_name": "White Bread", "category": "bakery", "price": 2.99, "quantity": 1},
                {"item_id": "milk", "item_name": "Whole Milk", "category": "dairy", "price": 3.49, "quantity": 1},
                {"item_id": "eggs", "item_name": "Large Eggs", "category": "dairy", "price": 4.99, "quantity": 1},
                {"item_id": "cheese", "item_name": "Cheddar Cheese", "category": "dairy", "price": 5.99, "quantity": 1}
            ]
        },
        {
            "transaction_id": "T005",
            "timestamp": "2024-01-01T18:20:00Z",
            "items": [
                {"item_id": "pasta", "item_name": "Spaghetti", "category": "pantry", "price": 1.99, "quantity": 2},
                {"item_id": "tomato_sauce", "item_name": "Marinara Sauce", "category": "pantry", "price": 2.99, "quantity": 1},
                {"item_id": "garlic", "item_name": "Garlic", "category": "produce", "price": 0.99, "quantity": 1}
            ]
        }
    ]
    return transactions

def test_analyze_endpoint():
    """Test the market basket analysis endpoint"""
    print("Testing /analyze endpoint...")
    
    transactions = create_test_transactions()
    
    payload = {
        "transactions": transactions,
        "min_support": 0.1,
        "min_confidence": 0.5,
        "algorithm": "apriori"
    }
    
    try:
        response = requests.post("http://localhost:8000/analyze", json=payload)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Analysis successful!")
            print(f"Total transactions: {result['total_transactions']}")
            print(f"Frequent itemsets found: {len(result['frequent_itemsets'])}")
            print(f"Association rules found: {len(result['association_rules'])}")
            
            # Print frequent itemsets
            print("\nFrequent Itemsets:")
            for itemset in result['frequent_itemsets']:
                print(f"  Items: {itemset['items']}, Support: {itemset['support']:.3f}")
            
            return result
        else:
            print(f"Error: {response.text}")
            return None
    
    except Exception as e:
        print(f"Request failed: {e}")
        return None

def test_recommendations_endpoint():
    """Test the recommendations endpoint"""
    print("\nTesting /recommendations endpoint...")
    
    # First analyze some data
    transactions = create_test_transactions()
    analyze_payload = {
        "transactions": transactions,
        "min_support": 0.1,
        "min_confidence": 0.5,
        "algorithm": "apriori"
    }
    
    # Analyze the data
    response = requests.post("http://localhost:8000/analyze", json=analyze_payload)
    if response.status_code != 200:
        print("Failed to analyze data first")
        return None
    
    # Now test recommendations
    recommendation_payload = {
        "current_items": ["bread", "milk"],
        "max_recommendations": 3
    }
    
    try:
        response = requests.post("http://localhost:8000/recommendations", json=recommendation_payload)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Recommendations successful!")
            print(f"Input items: {result['input_items']}")
            print(f"Recommendations count: {result['recommendation_count']}")
            
            # Print recommendations
            print("\nRecommendations:")
            for rec in result['recommendations']:
                print(f"  Item: {rec['item_id']}, Score: {rec['score']:.3f}")
            
            return result
        else:
            print(f"Error: {response.text}")
            return None
    
    except Exception as e:
        print(f"Request failed: {e}")
        return None

def test_frequent_itemsets_endpoint():
    """Test the frequent itemsets endpoint"""
    print("\nTesting /frequent-itemsets endpoint...")
    
    try:
        response = requests.get("http://localhost:8000/frequent-itemsets?min_support=0.1")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Frequent itemsets retrieved successfully!")
            print(f"Itemsets count: {len(result)}")
            
            # Print itemsets
            print("\nFrequent Itemsets:")
            for itemset in result:
                print(f"  Items: {itemset['items']}, Support: {itemset['support']:.3f}")
            
            return result
        else:
            print(f"Error: {response.text}")
            return None
    
    except Exception as e:
        print(f"Request failed: {e}")
        return None

def main():
    """Run all tests"""
    print("Starting Market Basket Analysis API tests...")
    print("=" * 50)
    
    # Wait a bit for the server to be ready
    time.sleep(2)
    
    # Test endpoints
    analyze_result = test_analyze_endpoint()
    
    if analyze_result:
        test_recommendations_endpoint()
        test_frequent_itemsets_endpoint()
    
    print("\n" + "=" * 50)
    print("Testing completed!")

if __name__ == "__main__":
    main()