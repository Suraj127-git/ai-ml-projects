# Image Classification for Products API

A FastAPI-based service for classifying product images using state-of-the-art computer vision models including EfficientNet, ResNet, and MobileNet architectures.

## Features

- **Multiple Model Support**: EfficientNet-B0, ResNet-50, MobileNet-V2, and Custom CNN
- **Product Categories**: Electronics, Clothing, Food, Books, Home & Kitchen, Sports, Beauty, Toys, Automotive, Health
- **Flexible Input**: Base64 encoded images or direct file uploads
- **Batch Processing**: Classify multiple images in a single request
- **Model Training**: Fine-tune models on custom datasets
- **Performance Evaluation**: Comprehensive model evaluation metrics
- **Configurable Parameters**: Top-k predictions, confidence thresholds, image sizes

## API Endpoints

### Core Classification
- `POST /classify` - Classify single image (base64)
- `POST /classify/batch` - Classify multiple images
- `POST /classify/upload` - Classify uploaded image file

### Model Management
- `GET /models` - Get available models information
- `POST /train` - Train/fine-tune models
- `POST /evaluate` - Evaluate model performance

### Utility
- `GET /health` - Health check and system status
- `GET /` - API information and available endpoints

## Installation

```bash
# Clone or create the project directory
cd image-classification-products

# Install dependencies
pip install -r requirements.txt

# Start the API server
python -m app.main
```

## Usage Examples

### Single Image Classification

```python
import requests
import base64
from PIL import Image
import io

# Load and encode image
image = Image.open("product.jpg")
buffered = io.BytesIO()
image.save(buffered, format="JPEG")
img_base64 = base64.b64encode(buffered.getvalue()).decode()

# Send classification request
response = requests.post("http://localhost:8008/classify", json={
    "image_data": img_base64,
    "model_type": "efficientnet_b0",
    "top_k": 5,
    "confidence_threshold": 0.1,
    "image_size": "small"
})

result = response.json()
print(f"Predictions: {result['predictions']}")
```

### Batch Classification

```python
# Multiple images
images = [img_base64_1, img_base64_2, img_base64_3]

response = requests.post("http://localhost:8008/classify/batch", json={
    "images": images,
    "model_type": "efficientnet_b0",
    "top_k": 3,
    "confidence_threshold": 0.1,
    "image_size": "small"
})

result = response.json()
print(f"Processed {result['total_images']} images")
print(f"Results: {result['results']}")
```

### File Upload Classification

```python
# Upload image file
with open("product.jpg", "rb") as f:
    files = {"file": f}
    data = {
        "model_type": "efficientnet_b0",
        "top_k": 5,
        "confidence_threshold": 0.1,
        "image_size": "small"
    }
    
    response = requests.post("http://localhost:8008/classify/upload", 
                           files=files, data=data)
    
    result = response.json()
    print(f"Predictions: {result['predictions']}")
```

## Model Information

### Available Models

| Model | Description | Parameters | Input Size |
|-------|-------------|------------|------------|
| EfficientNet-B0 | Lightweight and efficient CNN | 5.3M | 224×224 |
| ResNet-50 | Deep residual network | 25.6M | 224×224 |
| MobileNet-V2 | Optimized for mobile devices | 3.5M | 224×224 |
| Custom CNN | Simple convolutional network | 1M | 224×224 |

### Product Categories

- **Electronics**: Phones, laptops, cameras, accessories
- **Clothing**: Shirts, pants, dresses, shoes
- **Food**: Groceries, snacks, beverages
- **Books**: Fiction, non-fiction, textbooks
- **Home & Kitchen**: Appliances, furniture, decor
- **Sports**: Equipment, apparel, accessories
- **Beauty**: Cosmetics, skincare, haircare
- **Toys**: Games, puzzles, action figures
- **Automotive**: Parts, accessories, tools
- **Health**: Supplements, medical devices

## Configuration

### Image Sizes
- **Small**: 224×224 pixels (fast processing)
- **Medium**: 299×299 pixels (balanced quality/speed)
- **Large**: 512×512 pixels (highest quality)

### Model Parameters
- **top_k**: Number of top predictions to return (1-20)
- **confidence_threshold**: Minimum confidence score (0.0-1.0)
- **model_type**: Architecture to use for classification

## Training and Evaluation

### Model Training

```python
# Prepare training data
training_data = [
    {
        "image_data": base64_image_1,
        "label": "smartphone",
        "category": "electronics"
    },
    # ... more training samples
]

response = requests.post("http://localhost:8008/train", json={
    "training_data": training_data,
    "model_type": "custom_cnn",
    "epochs": 10,
    "learning_rate": 0.001,
    "validation_split": 0.2
})
```

### Model Evaluation

```python
# Prepare test data
test_data = [
    {
        "image_data": base64_image_1,
        "true_label": "smartphone",
        "true_category": "electronics"
    },
    # ... more test samples
]

response = requests.post("http://localhost:8008/evaluate", json={
    "test_data": test_data,
    "model_type": "efficientnet_b0"
})

result = response.json()
print(f"Accuracy: {result['accuracy']:.3f}")
print(f"F1-Score: {result['f1_score']:.3f}")
```

## Testing

Run the comprehensive test suite:

```bash
python test_api.py
```

The test suite includes:
- Health check verification
- Single image classification
- Batch classification
- Model information retrieval
- Sample data generation

## Performance

### Processing Times (approximate)
- **EfficientNet-B0**: ~50ms per image
- **ResNet-50**: ~100ms per image
- **MobileNet-V2**: ~30ms per image
- **Custom CNN**: ~20ms per image

*Note: Actual processing times depend on hardware and image size.*

## Error Handling

The API provides detailed error messages for common issues:

- **400 Bad Request**: Invalid input data or parameters
- **404 Not Found**: Model or endpoint not available
- **500 Internal Server Error**: Processing or model errors
- **503 Service Unavailable**: Model not loaded or unavailable

## Development

### Project Structure
```
image-classification-products/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   ├── model.py          # Model logic and architectures
│   └── schemas.py        # Pydantic data models
├── requirements.txt      # Dependencies
├── test_api.py          # API tests
└── README.md            # Documentation
```

### Adding New Models

1. Add model type to `schemas.py`
2. Implement model loading in `model.py`
3. Update preprocessing if needed
4. Add model information to documentation

### Custom Categories

To add new product categories:

1. Update `ProductCategory` enum in `schemas.py`
2. Retrain models with new categories
3. Update model output layers
4. Test with new category data

## Deployment

### Production Setup

```bash
# Install production dependencies
pip install -r requirements.txt

# Start with production server
uvicorn app.main:app --host 0.0.0.0 --port 8008 --workers 4
```

### Docker (Optional)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8008"]
```

## License

This project is part of the AI/ML projects collection and follows the same licensing terms.

## Support

For issues and questions:
- Check the error messages and logs
- Verify model files are properly loaded
- Ensure image formats are supported (JPEG, PNG, etc.)
- Check system resources and GPU availability