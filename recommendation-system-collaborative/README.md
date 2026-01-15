# Collaborative Recommendation System

A sophisticated recommendation system built with collaborative filtering techniques and matrix factorization algorithms. This system provides personalized recommendations using multiple algorithms including Non-negative Matrix Factorization (NMF), user-based collaborative filtering, item-based collaborative filtering, and hybrid approaches.

## Features

- **Multiple Recommendation Algorithms**:
  - Matrix Factorization (NMF)
  - User-based Collaborative Filtering
  - Item-based Collaborative Filtering
  - Hybrid approaches combining multiple methods

- **RESTful API** built with FastAPI
- **Scalable Architecture** supporting large datasets
- **Comprehensive Evaluation Metrics** (Precision@K, Recall@K, F1-Score)
- **Real-time Recommendations** with confidence scoring
- **Batch Processing** for efficient data ingestion
- **User and Item Management** with metadata support

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd recommendation-system-collaborative
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the API server:
```bash
python -m app.main
```

The API will be available at `http://localhost:8000`

### Basic Usage

1. **Add a user**:
```bash
curl -X POST "http://localhost:8000/users" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "age": 25,
    "gender": "male",
    "location": "New York",
    "preferences": {"genres": ["action", "comedy"]}
  }'
```

2. **Add an item**:
```bash
curl -X POST "http://localhost:8000/items" \
  -H "Content-Type: application/json" \
  -d '{
    "item_id": "item_001",
    "item_name": "The Matrix",
    "category": "movie",
    "tags": ["action", "sci-fi"],
    "metadata": {"year": 1999, "rating": 4.5}
  }'
```

3. **Add a rating**:
```bash
curl -X POST "http://localhost:8000/ratings" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "item_id": "item_001",
    "rating": 4.5
  }'
```

4. **Train the model**:
```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "matrix_factorization",
    "hyperparameters": {
      "n_components": 50,
      "max_iter": 200
    }
  }'
```

5. **Get recommendations**:
```bash
curl -X POST "http://localhost:8000/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "algorithm": "matrix_factorization",
    "num_recommendations": 10
  }'
```

## API Endpoints

### System Information
- `GET /` - API information and available algorithms
- `GET /health` - Health check and system statistics
- `GET /stats` - Detailed system statistics

### Data Management
- `POST /users` - Add a new user
- `POST /items` - Add a new item
- `POST /ratings` - Add a rating
- `POST /batch-ratings` - Add multiple ratings in batch
- `GET /users/{user_id}/ratings` - Get all ratings for a user
- `GET /items/{item_id}/ratings` - Get all ratings for an item

### Model Operations
- `POST /train` - Train the recommendation model
- `POST /recommendations` - Generate recommendations for a user
- `POST /evaluate` - Evaluate model performance
- `DELETE /reset` - Reset the recommendation engine

## Algorithms

### Matrix Factorization (NMF)
Uses Non-negative Matrix Factorization to decompose the user-item interaction matrix into user and item feature matrices. This approach is particularly effective for:
- Discovering latent features in user preferences
- Handling sparse data efficiently
- Providing interpretable recommendations

### Collaborative Filtering
- **User-based**: Finds similar users and recommends items they liked
- **Item-based**: Finds similar items to those the user has rated highly
- **Hybrid**: Combines multiple approaches for improved accuracy

## Configuration

### Training Hyperparameters

```json
{
  "algorithm": "matrix_factorization",
  "hyperparameters": {
    "n_components": 50,      // Number of latent factors
    "max_iter": 200,         // Maximum iterations for convergence
    "init": "random",        // Initialization method
    "solver": "cd"           // Solver algorithm
  }
}
```

### Algorithm Options

- `matrix_factorization`: Pure matrix factorization approach
- `collaborative_filtering`: User-based collaborative filtering
- `item_based`: Item-based collaborative filtering  
- `hybrid`: Combines multiple methods

## Performance Metrics

The system provides comprehensive evaluation metrics:

- **Precision@K**: Proportion of recommended items that are relevant
- **Recall@K**: Proportion of relevant items that are recommended
- **F1-Score**: Harmonic mean of precision and recall

## Testing

Run the comprehensive test suite:

```bash
python test_api.py
```

This will:
- Test all API endpoints
- Create sample users, items, and ratings
- Train the model with different algorithms
- Generate recommendations
- Evaluate model performance

## Validation

Validate the installation and setup:

```bash
python validate.py
```

This checks:
- All required dependencies
- Schema definitions
- Model functionality
- FastAPI app configuration

## Data Format

### User Schema
```json
{
  "user_id": "string",
  "age": "integer (optional)",
  "gender": "string (optional)",
  "location": "string (optional)",
  "preferences": "object (optional)"
}
```

### Item Schema
```json
{
  "item_id": "string",
  "item_name": "string",
  "category": "string",
  "tags": "array of strings (optional)",
  "metadata": "object (optional)"
}
```

### Rating Schema
```json
{
  "user_id": "string",
  "item_id": "string",
  "rating": "number (1.0 to 5.0)",
  "timestamp": "string (ISO 8601, optional)"
}
```

## Best Practices

1. **Data Quality**: Ensure ratings are within the 1.0-5.0 range
2. **Minimum Data**: Train with at least 10 ratings for meaningful results
3. **Sparsity**: The system handles sparse data well, but more data improves recommendations
4. **Algorithm Selection**: Start with matrix factorization, then experiment with hybrid approaches
5. **Hyperparameter Tuning**: Adjust `n_components` based on your dataset size and complexity

## Troubleshooting

### Common Issues

1. **"Insufficient data" error**: Add more ratings (minimum 10 required)
2. **"User not found" error**: Ensure users are added before generating recommendations
3. **"Model not trained" error**: Train the model before generating recommendations
4. **Poor recommendations**: Try different algorithms or adjust hyperparameters

### Performance Optimization

- Use batch operations for adding multiple ratings
- Consider the sparsity of your dataset
- Adjust `n_components` based on dataset size
- Use appropriate algorithms for your data characteristics

## Architecture

```
recommendation-system-collaborative/
├── app/
│   ├── __init__.py          # Package initialization
│   ├── main.py              # FastAPI application
│   ├── model.py             # Recommendation engine implementation
│   └── schemas.py           # Pydantic data models
├── requirements.txt         # Dependencies
├── test_api.py             # Comprehensive test suite
├── validate.py             # Validation script
└── README.md              # Documentation
```

## License

This project is part of the AI/ML Projects collection and follows the same licensing terms.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Run the validation script
3. Review the test suite for examples
4. Check API documentation at `/docs` endpoint