#!/usr/bin/env python3
"""
Market Basket Analysis Validation Script
Validates the market basket analysis implementation
"""

import sys
import os
from typing import List, Dict

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

try:
    from schemas import Transaction, TransactionItem, FrequentItemset
    from model import MarketBasketAnalyzer
    print("✓ Successfully imported all modules")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

def create_test_data():
    """Create test transaction data"""
    transactions = [
        Transaction(
            transaction_id="T001",
            items=[
                TransactionItem(item_id="bread", item_name="White Bread", price=2.99),
                TransactionItem(item_id="milk", item_name="Whole Milk", price=3.49),
                TransactionItem(item_id="eggs", item_name="Large Eggs", price=4.99)
            ]
        ),
        Transaction(
            transaction_id="T002",
            items=[
                TransactionItem(item_id="bread", item_name="White Bread", price=2.99),
                TransactionItem(item_id="butter", item_name="Salted Butter", price=4.49)
            ]
        ),
        Transaction(
            transaction_id="T003",
            items=[
                TransactionItem(item_id="milk", item_name="Whole Milk", price=3.49),
                TransactionItem(item_id="cereal", item_name="Corn Flakes", price=3.99),
                TransactionItem(item_id="bananas", item_name="Bananas", price=1.99)
            ]
        )
    ]
    return transactions

def test_basic_functionality():
    """Test basic functionality"""
    print("\n=== Testing Basic Functionality ===")
    
    try:
        # Create analyzer
        analyzer = MarketBasketAnalyzer()
        print("✓ MarketBasketAnalyzer created successfully")
        
        # Load transactions
        transactions = create_test_data()
        analyzer.load_transactions(transactions)
        print(f"✓ Loaded {len(transactions)} transactions")
        
        # Perform analysis
        analyzer.analyze(min_support=0.1)
        print(f"✓ Analysis completed, found {len(analyzer.frequent_itemsets)} frequent itemsets")
        
        # Check results
        if len(analyzer.frequent_itemsets) > 0:
            print("✓ Frequent itemsets found:")
            for itemset in analyzer.frequent_itemsets:
                print(f"  - {itemset.items}: support={itemset.support:.3f}")
        
        # Test recommendations
        recommendations = analyzer.get_recommendations(["bread"], top_k=3)
        print(f"✓ Generated {len(recommendations)} recommendations")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_edge_cases():
    """Test edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    try:
        analyzer = MarketBasketAnalyzer()
        
        # Test empty transactions
        analyzer.load_transactions([])
        analyzer.analyze(min_support=0.1)
        print("✓ Handled empty transactions")
        
        # Test single transaction
        single_transaction = [
            Transaction(
                transaction_id="T001",
                items=[TransactionItem(item_id="apple", item_name="Apple", price=1.0)]
            )
        ]
        analyzer.load_transactions(single_transaction)
        analyzer.analyze(min_support=0.1)
        print("✓ Handled single transaction")
        
        # Test very high support threshold
        multiple_transactions = create_test_data()
        analyzer.load_transactions(multiple_transactions)
        analyzer.analyze(min_support=0.9)  # Very high threshold
        print(f"✓ High support threshold handled, found {len(analyzer.frequent_itemsets)} itemsets")
        
        return True
        
    except Exception as e:
        print(f"✗ Edge case test failed: {e}")
        return False

def test_data_validation():
    """Test data validation"""
    print("\n=== Testing Data Validation ===")
    
    try:
        # Test schema validation
        item = TransactionItem(item_id="test", item_name="Test Item", price=9.99)
        transaction = Transaction(transaction_id="T001", items=[item])
        print("✓ Schema validation works")
        
        # Test with missing optional fields
        minimal_item = TransactionItem(item_id="minimal", item_name="Minimal Item")
        minimal_transaction = Transaction(transaction_id="T002", items=[minimal_item])
        print("✓ Optional fields handled correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Data validation test failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("Market Basket Analysis Validation")
    print("=" * 40)
    
    tests = [
        test_basic_functionality,
        test_edge_cases,
        test_data_validation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== Validation Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Market Basket Analysis implementation is ready.")
        return 0
    else:
        print("✗ Some tests failed. Please review the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())