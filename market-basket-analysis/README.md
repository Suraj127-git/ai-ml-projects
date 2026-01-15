# Market Basket Analysis with Association Rules

A FastAPI-based market basket analysis system that discovers frequent itemsets and generates association rules from transaction data. This implementation uses the Apriori algorithm to identify patterns in customer purchasing behavior.

## Features

- **Frequent Itemset Mining**: Discover items that frequently appear together in transactions
- **Association Rule Generation**: Generate rules like "If customer buys bread, they are likely to buy milk"
- **Product Recommendations**: Get personalized product recommendations based on current basket
- **Multiple Algorithm Support**: Framework ready for Apriori, FP-Growth, and ECLAT algorithms
- **RESTful API**: Easy-to-use HTTP endpoints for analysis and recommendations

## Installation

```bash
# Clone or download the project
cd market-basket-analysis

# Install dependencies
pip install -r requirements.txt

# Run the API server
python -m app.main
```

## API Endpoints

### Health Check
```http
GET /health
```
Returns the health status of the API.

### Analyze Transactions
```http
POST /analyze
```
Performs market basket analysis on provided transaction data.

**Request Body:**
```json
{
  "transactions": [
    {
      "transaction_id": "T001",
      "items": [
        {"item_id": "bread", "item_name": "White Bread", "price": 2.99},
        {"item_id": "milk", "item_name": "Whole Milk", "price": 3.49}
      ]
    }
  ],
  "min_support": 0.01,
  "min_confidence": 0.5,
  "algorithm": "apriori"
}
```

### Get Recommendations
```http
POST /recommendations
```
Get product recommendations based on current items in basket.

**Request Body:**
```json
{
  "items": ["bread", "milk"],
  "top_k": 5
}
```

### Get Frequent Itemsets
```http
GET /frequent-itemsets?min_support=0.1&min_length=2
```
Retrieve frequent itemsets with optional filtering.

### Get Association Rules
```http
GET /association-rules?min_confidence=0.5&min_lift=1.0
```
Retrieve association rules with optional filtering.

### Get Statistics
```http
GET /statistics
```
Get analysis statistics including transaction count and itemset information.

## Usage Examples

### Basic Analysis

```python
import requests

# Prepare transaction data
transactions = [
    {
        "transaction_id": "T001",
        "items": [
            {"item_id": "bread", "item_name": "White Bread", "price": 2.99},
            {"item_id": "milk", "item_name": "Whole Milk", "price": 3.49},
            {"item_id": "eggs", "item_name": "Large Eggs", "price": 4.99}
        ]
    },
    # Add more transactions...
]

# Analyze transactions
response = requests.post("http://localhost:8000/analyze", json={
    "transactions": transactions,
    "min_support": 0.1,
    "min_confidence": 0.5,
    "algorithm": "apriori"
})

result = response.json()
print(f"Found {len(result['frequent_itemsets'])} frequent itemsets")
print(f"Found {len(result['association_rules'])} association rules")
```

### Getting Recommendations

```python
# Get recommendations for items in basket
response = requests.post("http://localhost:8000/recommendations", json={
    "items": ["bread", "milk"],
    "top_k": 3
})

recommendations = response.json()
for rec in recommendations['recommendations']:
    print(f"Recommended: {rec['item_id']} (score: {rec['score']:.3f})")
```

## Data Models

### Transaction
```python
class Transaction(BaseModel):
    transaction_id: str
    items: List[TransactionItem]
    timestamp: Optional[datetime] = None
    customer_id: Optional[str] = None
    total_amount: Optional[float] = None
```

### TransactionItem
```python
class TransactionItem(BaseModel):
    item_id: str
    item_name: str
    category: Optional[str] = None
    price: Optional[float] = None
    quantity: Optional[int] = 1
```

### FrequentItemset
```python
class FrequentItemset(BaseModel):
    items: List[str]
    support: float
    count: int
    length: int
```

### AssociationRule
```python
class AssociationRule(BaseModel):
    antecedent: List[str]  # "if" part of the rule
    consequent: List[str]  # "then" part of the rule
    support: float
    confidence: float
    lift: float
```

## Algorithms

### Apriori Algorithm
The current implementation uses the Apriori algorithm for frequent itemset mining:

1. **Candidate Generation**: Generate candidate itemsets of length k from frequent itemsets of length k-1
2. **Support Counting**: Count the occurrence of each candidate in the transaction database
3. **Frequent Itemset Selection**: Select candidates that meet the minimum support threshold
4. **Rule Generation**: Generate association rules from frequent itemsets

### Future Enhancements
- **FP-Growth**: More efficient for large datasets
- **ECLAT**: Better for datasets with many items
- **Constraint-based Mining**: Support for user-defined constraints

## Testing

Run the validation script to test the implementation:

```bash
python validate.py
```

Run the API tests:

```bash
python test_api.py
```

## Configuration

### Minimum Support
- **Definition**: Minimum frequency of itemsets to be considered frequent
- **Range**: 0.0 to 1.0
- **Default**: 0.01 (1%)
- **Recommendation**: Start with 0.01-0.05 for most datasets

### Minimum Confidence
- **Definition**: Minimum probability that the rule holds
- **Range**: 0.0 to 1.0
- **Default**: 0.5 (50%)
- **Recommendation**: Start with 0.3-0.7 depending on domain

## Performance Considerations

- **Dataset Size**: Optimized for datasets with thousands of transactions
- **Memory Usage**: Efficient memory usage for itemset storage
- **Scalability**: Framework ready for distributed processing

## Use Cases

1. **Retail**: Product placement optimization, cross-selling strategies
2. **E-commerce**: Personalized recommendations, bundle creation
3. **Grocery**: Understanding customer shopping patterns
4. **Marketing**: Targeted promotions, customer segmentation

## Limitations

- Current implementation focuses on single-item recommendations
- Association rule generation is simplified
- No support for temporal patterns (yet)
- Basic implementation of Apriori algorithm

## Contributing

This is part of a comprehensive ML project collection. The implementation follows API-first design principles and can be extended with:

- Advanced association rule algorithms
- Temporal pattern mining
- Multi-level association rules
- Constraint-based mining
- Integration with recommendation systems