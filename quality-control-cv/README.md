# Quality Control Computer Vision System

An AI-powered quality control system that uses computer vision and deep learning (CNN/ResNet) to automatically detect defects in manufactured products. The system provides real-time quality assessment with detailed defect analysis and recommendations.

## Features

- **Computer Vision**: CNN and ResNet-based defect detection
- **Multiple Defect Types**: Supports 8 different defect categories
- **Product Categories**: Quality standards for 8 product types
- **Real-time Processing**: Fast image analysis and defect detection
- **Batch Processing**: Inspect multiple products simultaneously
- **Quality Metrics**: Comprehensive quality scoring and analytics
- **Recommendations**: AI-powered quality improvement suggestions

## Supported Defect Types

- **Scratch**: Surface scratches and abrasions
- **Crack**: Structural cracks and fractures
- **Dent**: Surface deformations and dents
- **Discoloration**: Color variations and staining
- **Missing Part**: Absent components or features
- **Dimensional Error**: Size and shape deviations
- **Surface Defect**: Surface texture irregularities
- **Contamination**: Foreign particles or substances

## Supported Product Categories

- **Electronics**: Consumer electronics and components
- **Automotive**: Vehicle parts and assemblies
- **Textiles**: Fabrics and clothing materials
- **Food**: Food products and packaging
- **Pharmaceutical**: Medical products and devices
- **Metal**: Metal components and structures
- **Plastic**: Plastic products and parts
- **Ceramic**: Ceramic and porcelain items

## API Endpoints

### Core Endpoints

- `POST /inspect-product` - Inspect a single product
- `POST /inspect-batch` - Inspect multiple products
- `POST /upload-image` - Upload and inspect image
- `GET /model-info` - Get model information
- `GET /quality-standards` - Get quality standards
- `GET /statistics` - Get processing statistics
- `POST /generate-sample-data` - Generate test data
- `GET /health` - Health check endpoint

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Start the API Server

```bash
python -m app.main
```

The server will start on `http://localhost:8003`

### Example Usage

#### Single Product Inspection

```python
import requests
import base64

# Read and encode image
with open("product_image.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

# Prepare request
request_data = {
    "product_id": "laptop_001",
    "product_name": "Gaming Laptop Pro",
    "category": "electronics",
    "image_data": f"data:image/jpeg;base64,{encoded_string}",
    "image_format": "jpeg",
    "batch_id": "batch_001",
    "manufacturing_date": "2024-01-15T10:00:00"
}

# Send request
response = requests.post("http://localhost:8003/inspect-product", json=request_data)
result = response.json()

print(f"Quality Status: {result['status']}")
print(f"Quality Score: {result['quality_score']:.3f}")
print(f"Defects Found: {len(result['defects_detected'])}")
```

#### Batch Inspection

```python
# Prepare multiple products
products = [
    {
        "product_id": "product_001",
        "product_name": "Smartphone Screen",
        "category": "electronics",
        "image_data": "base64_encoded_image_data_1",
        "image_format": "jpeg",
        "batch_id": "batch_002"
    },
    {
        "product_id": "product_002",
        "product_name": "Car Door Panel",
        "category": "automotive",
        "image_data": "base64_encoded_image_data_2",
        "image_format": "jpeg",
        "batch_id": "batch_002"
    }
]

request_data = {
    "products": products,
    "batch_config": {
        "parallel_processing": True,
        "quality_threshold": 0.8
    }
}

response = requests.post("http://localhost:8003/inspect-batch", json=request_data)
result = response.json()

print(f"Batch ID: {result['batch_id']}")
print(f"Total Products: {result['total_products']}")
print(f"Passed: {result['passed_products']}")
print(f"Failed: {result['failed_products']}")
print(f"Defect Rate: {result['batch_summary']['defect_rate']:.3f}")
```

#### Image Upload

```python
import requests

# Upload image file
with open("product.jpg", "rb") as f:
    files = {'file': ('product.jpg', f, 'image/jpeg')}
    data = {
        'product_id': 'product_123',
        'product_name': 'Test Product',
        'category': 'electronics',
        'batch_id': 'batch_003'
    }
    
    response = requests.post("http://localhost:8003/upload-image", files=files, data=data)
    result = response.json()

print(f"Inspection Result: {result['status']}")
print(f"Quality Score: {result['quality_score']:.3f}")
```

## Computer Vision Architecture

### CNN Model
The custom CNN architecture includes:
- **Feature Extraction**: 4 convolutional layers with max pooling
- **Classification Head**: Dense layers for defect classification
- **Multi-task Learning**: Simultaneous defect detection and quality scoring
- **Transfer Learning**: Pre-trained ResNet50 backbone

### Image Processing Pipeline
1. **Image Preprocessing**: Resize, normalize, and augment
2. **Feature Extraction**: CNN/ResNet feature extraction
3. **Defect Detection**: Multi-class defect classification
4. **Quality Scoring**: Overall quality assessment
5. **Post-processing**: Threshold application and NMS

## Quality Assessment Algorithm

### Defect Detection
```python
def detect_defects(self, image, category):
    # Get predictions from classifier
    defect_scores, quality_score, model_info = self.classifier.predict(image)
    
    # Apply detection threshold
    defects = []
    for i, score in enumerate(defect_scores):
        if score > 0.3:  # Detection threshold
            defect = DefectDetection(
                defect_type=defect_types[i],
                confidence=float(score),
                location=calculate_defect_location(image, score),
                severity=calculate_severity(score),
                description=generate_defect_description(defect_types[i], category)
            )
            defects.append(defect)
    
    return defects, quality_score
```

### Quality Status Determination
The system determines quality status based on:
- **Critical Defects**: High-confidence defects that automatically fail
- **Defect Count**: Number of defects vs. category-specific limits
- **Quality Score**: Overall quality assessment score
- **Standards Compliance**: Category-specific quality standards

## Quality Standards

Each product category has specific quality standards:

| Category | Max Defects | Min Quality Score | Critical Threshold |
|----------|-------------|-------------------|-------------------|
| Electronics | 2 | 0.85 | 0.80 |
| Automotive | 1 | 0.90 | 0.85 |
| Textiles | 3 | 0.80 | 0.75 |
| Food | 1 | 0.92 | 0.90 |
| Pharmaceutical | 0 | 0.95 | 0.95 |
| Metal | 2 | 0.88 | 0.80 |
| Plastic | 3 | 0.82 | 0.75 |
| Ceramic | 2 | 0.85 | 0.80 |

## Performance Metrics

The system tracks comprehensive metrics:

- **Processing Speed**: Average time per inspection
- **Detection Accuracy**: Defect detection precision/recall
- **Quality Score Distribution**: Statistical analysis of quality scores
- **Defect Rate**: Percentage of products with defects
- **Batch Performance**: Multi-product processing efficiency

## Testing

Run the comprehensive test suite:

```bash
python test_api.py
```

Tests include:
- Health check verification
- Single product inspection
- Batch processing
- Image upload functionality
- Model information retrieval
- Quality standards validation
- Statistics reporting
- Sample data generation

## Configuration

Key configuration parameters:

- **Detection Threshold**: 0.3 (30% confidence)
- **Image Size**: 224x224 pixels
- **Model Version**: 1.0.0
- **Device**: Auto-detect (CUDA if available)

## Data Requirements

Input requirements:
- **Image Format**: JPEG, PNG supported
- **Image Size**: Minimum 224x224 pixels
- **Encoding**: Base64 encoded
- **Metadata**: Product ID, name, category
- **Optional**: Batch ID, manufacturing date

## Model Training

The system supports custom model training:

1. **Data Collection**: Gather labeled defect images
2. **Annotation**: Mark defect locations and types
3. **Training**: Fine-tune CNN/ResNet models
4. **Validation**: Test on held-out validation set
5. **Deployment**: Update model in production

## License

This project is part of the AI/ML Projects collection.