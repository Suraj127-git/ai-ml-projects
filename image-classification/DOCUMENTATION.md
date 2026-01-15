# 🖼️ Image Classification - Comprehensive Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Technical Implementation](#technical-implementation)
3. [Architectural Documentation](#architectural-documentation)
4. [Project Structure](#project-structure)
5. [Learning Pathways](#learning-pathways)
6. [Setup and Usage](#setup-and-usage)
7. [API Reference](#api-reference)
8. [Performance Metrics](#performance-metrics)

---

## Project Overview

### Purpose and Goals
The Image Classification service is a deep learning microservice that classifies images into predefined categories using Convolutional Neural Networks (CNNs). It supports multiple state-of-the-art architectures including custom CNNs, ResNet, and MobileNet, with capabilities for transfer learning and model training on custom datasets.

### Business Problem
- **Problem**: Manual image categorization is time-consuming, inconsistent, and doesn't scale. Organizations process millions of images daily (e-commerce products, medical scans, security footage, social media content).
- **Solution**: Automated image classification using deep learning models that can categorize images with human-level accuracy at scale.
- **Impact**: 
  - Process thousands of images per minute
  - Consistent classification quality
  - 99%+ reduction in manual labeling time
  - Enable real-time image-based applications
  - Support content moderation, product categorization, quality control

### Expected Outcomes
- **Primary Metrics**:
  - Classification Accuracy: >90% (CIFAR-10)
  - Top-5 Accuracy: >95%
  - Inference Speed: <200ms per image
  - Throughput: 50+ images/second
  - Model Size: <100MB (optimized models)

- **Success Criteria**:
  - Accurately classify images across multiple categories
  - Support batch processing for high throughput
  - Provide confidence scores for predictions
  - Enable model retraining with custom datasets
  - Deploy with minimal latency (<200ms)

---

## Technical Implementation

### Technology Stack

#### Core Technologies

1. **TensorFlow/Keras** (v2.13+)
   - High-level neural network API
   - Pre-trained model zoo (ImageNet weights)
   - GPU acceleration support
   - Easy model saving/loading
   - Extensive layer library

2. **FastAPI** (v0.104.1)
   - Async image processing
   - WebSocket support for streaming
   - Automatic API documentation
   - File upload handling
   - Base64 image encoding support

3. **OpenCV** (v4.8+)
   - Image preprocessing
   - Resize, crop, normalize operations
   - Color space conversions
   - Image augmentation

4. **Pillow (PIL)** (v10.0+)
   - Image I/O operations
   - Format conversions
   - Basic transformations

5. **NumPy** (v1.24+)
   - Array operations
   - Image tensor manipulation
   - Batch processing

### Technology Alternatives

| Component | Current Choice | Alternative 1 | Alternative 2 | Why Current Choice |
|-----------|---------------|---------------|---------------|-------------------|
| **DL Framework** | TensorFlow/Keras | PyTorch | MXNet | Production-ready, extensive pretrained models, easier deployment |
| **Model Architecture** | ResNet/MobileNet | EfficientNet | Vision Transformer | Balance of accuracy and speed, proven performance |
| **Image Processing** | OpenCV + PIL | torchvision | scikit-image | Performance, comprehensive features, industry standard |
| **Web Framework** | FastAPI | Flask | Django | Async support, automatic docs, high performance |
| **Model Format** | HDF5 (.h5) | ONNX | SavedModel | Native TensorFlow, easy to use, compatible |

### CNN Architectures Explained

#### 1. Custom CNN
```
Architecture:
Input (32x32x3)
    ↓
Conv2D(32 filters, 3x3) + ReLU + BatchNorm
    ↓
Conv2D(32 filters, 3x3) + ReLU + BatchNorm
    ↓
MaxPooling2D(2x2)
    ↓
Dropout(0.25)
    ↓
Conv2D(64 filters, 3x3) + ReLU + BatchNorm
    ↓
Conv2D(64 filters, 3x3) + ReLU + BatchNorm
    ↓
MaxPooling2D(2x2)
    ↓
Dropout(0.25)
    ↓
Flatten
    ↓
Dense(512) + ReLU
    ↓
Dropout(0.5)
    ↓
Dense(num_classes) + Softmax
    ↓
Output (probabilities)

Parameters: ~1.2M
Training Time: 30-60 min (CPU)
Accuracy: ~75-80% (CIFAR-10)
```

**Pros**:
- Small model size (~5MB)
- Fast inference (<50ms)
- Easy to understand
- Good for learning

**Cons**:
- Lower accuracy than SOTA
- Limited depth
- May underfit complex data

#### 2. ResNet-18
```
Architecture:
Input (224x224x3)
    ↓
Conv2D(64, 7x7, stride=2)
    ↓
MaxPooling(3x3, stride=2)
    ↓
Residual Block 1 (64 filters) x2
    ↓
Residual Block 2 (128 filters) x2
    ↓
Residual Block 3 (256 filters) x2
    ↓
Residual Block 4 (512 filters) x2
    ↓
GlobalAveragePooling
    ↓
Dense(num_classes) + Softmax

Residual Block:
    input
    ↓
    Conv → BN → ReLU → Conv → BN
    ↓                           ↓
    └─────────── + ─────────────┘
           ↓
          ReLU

Parameters: ~11.7M
Training Time: 2-4 hours (GPU)
Accuracy: ~90-92% (CIFAR-10)
```

**Pros**:
- Solves vanishing gradient problem
- Can train very deep networks (50-200 layers)
- Excellent accuracy
- Transfer learning from ImageNet

**Cons**:
- Larger model size (~45MB)
- Slower inference (~150ms)
- Requires more GPU memory

#### 3. MobileNet
```
Architecture:
Input (224x224x3)
    ↓
Conv2D(32, 3x3, stride=2)
    ↓
Depthwise Separable Conv Block x13
    ↓
GlobalAveragePooling
    ↓
Dense(num_classes) + Softmax

Depthwise Separable Conv:
    Depthwise Conv (3x3)
        ↓
    Pointwise Conv (1x1)

Parameters: ~4.2M
Training Time: 1-2 hours (GPU)
Accuracy: ~88-90% (CIFAR-10)
```

**Pros**:
- Lightweight (~17MB)
- Fast inference (~80ms)
- Mobile-optimized
- Good accuracy/speed tradeoff

**Cons**:
- Slightly lower accuracy than ResNet
- More complex architecture
- Depthwise convolutions harder to optimize

### Code Function Explanations

#### 1. Model Architecture (`model.py`)

```python
class ImageClassificationModel:
    """
    Comprehensive image classification model with multiple architectures
    """
    
    def __init__(self):
        """
        Initialize model components:
        - model: Keras model (None until trained/loaded)
        - input_shape: Image dimensions (height, width, channels)
        - num_classes: Number of categories
        - class_names: List of category labels
        - model_name: Architecture type
        """
        self.model = None
        self.input_shape = (32, 32, 3)
        self.num_classes = 10
        self.class_names = []
        self.model_name = "cnn_custom"
    
    def build_custom_cnn(self, input_shape, num_classes):
        """
        Build custom CNN architecture
        
        Design Principles:
        1. Progressive feature extraction (32 → 64 filters)
        2. Regularization (BatchNorm, Dropout)
        3. Spatial reduction (MaxPooling)
        4. Dense classifier head
        
        Returns: Compiled Keras model
        """
        model = models.Sequential([
            # First conv block
            layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            
            layers.Conv2D(32, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second conv block
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Classifier
            layers.Flatten(),
            layers.Dense(512),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def build_resnet18(self, input_shape, num_classes):
        """
        Build ResNet-18 architecture
        
        Key Innovation: Residual connections
        - Skip connections allow gradients to flow
        - Enables training of very deep networks
        - Identity mapping prevents degradation
        
        Transfer Learning:
        1. Load ImageNet pre-trained weights
        2. Freeze early layers (feature extraction)
        3. Fine-tune later layers
        4. Replace classification head
        
        Returns: Compiled Keras model
        """
        base_model = applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling='avg'
        )
        
        # Freeze base layers
        for layer in base_model.layers[:-10]:
            layer.trainable = False
        
        # Add custom head
        x = base_model.output
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        predictions = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=base_model.input, outputs=predictions)
        
        return model
    
    def train_model(self, dataset_name='cifar10', model_name='cnn_custom',
                   epochs=50, batch_size=64, learning_rate=0.001):
        """
        Train image classification model
        
        Process:
        1. Load dataset (CIFAR-10, CIFAR-100, or custom)
        2. Preprocess images (normalize, augment)
        3. Build model architecture
        4. Compile with optimizer and loss
        5. Set up callbacks (early stopping, checkpointing)
        6. Train with validation split
        7. Evaluate on test set
        8. Save model and metadata
        
        Data Augmentation:
        - Random horizontal flip
        - Random rotation (±15°)
        - Random zoom (±10%)
        - Random shift (±10%)
        - Purpose: Prevent overfitting, improve generalization
        
        Training Strategy:
        - Optimizer: Adam (adaptive learning rate)
        - Loss: Categorical crossentropy
        - Metrics: Accuracy, top-5 accuracy
        - Early stopping: patience=10 epochs
        - Learning rate reduction: factor=0.5, patience=5
        
        Returns: Training history and metrics
        """
        # Load data
        if dataset_name == 'cifar10':
            (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
            self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                               'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Normalize pixel values [0, 255] → [0, 1]
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        # One-hot encode labels
        y_train = to_categorical(y_train, self.num_classes)
        y_test = to_categorical(y_test, self.num_classes)
        
        # Build model
        if model_name == 'cnn_custom':
            self.model = self.build_custom_cnn(self.input_shape, self.num_classes)
        elif model_name == 'resnet18':
            self.model = self.build_resnet18((224, 224, 3), self.num_classes)
        
        # Compile
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
            ModelCheckpoint('best_model.h5', save_best_only=True)
        ]
        
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1
        )
        datagen.fit(X_train)
        
        # Train
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_acc, test_top5 = self.model.evaluate(X_test, y_test)
        
        return {
            'test_accuracy': float(test_acc),
            'test_loss': float(test_loss),
            'test_top5_accuracy': float(test_top5),
            'history': history.history
        }
    
    def predict(self, image_base64, confidence_threshold=0.5, top_k=5):
        """
        Classify single image
        
        Process:
        1. Decode base64 string to image
        2. Preprocess image (resize, normalize)
        3. Add batch dimension
        4. Model inference
        5. Get top-K predictions
        6. Filter by confidence threshold
        7. Format response
        
        Preprocessing:
        - Decode base64 → PIL Image
        - Convert to RGB (if needed)
        - Resize to model input size
        - Convert to numpy array
        - Normalize [0, 255] → [0, 1]
        - Expand dims: (H, W, C) → (1, H, W, C)
        
        Returns: Predictions with class names, probabilities
        """
        # Decode image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize
        image = image.resize((self.input_shape[0], self.input_shape[1]))
        
        # Convert to array and normalize
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Predict
        start_time = time.time()
        predictions = self.model.predict(img_array, verbose=0)[0]
        inference_time = time.time() - start_time
        
        # Get top-K predictions
        top_indices = np.argsort(predictions)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            confidence = float(predictions[idx])
            if confidence >= confidence_threshold:
                results.append({
                    'class_name': self.class_names[idx],
                    'class_id': int(idx),
                    'confidence': confidence
                })
        
        return {
            'predictions': results,
            'processing_time': inference_time,
            'model_name': self.model_name
        }
```

#### 2. API Endpoints (`main.py`)

```python
@app.post("/predict", response_model=ImageClassificationResponse)
async def classify_image(request: ImageClassificationRequest):
    """
    Classify single image
    
    Input:
    - image_base64: Base64-encoded image string
    - confidence_threshold: Minimum confidence (default: 0.5)
    - top_k: Number of top predictions (default: 5)
    
    Output:
    - predictions: List of class predictions with confidence
    - model_name: Model architecture used
    - processing_time: Inference time in seconds
    
    Example Request:
    {
        "image_base64": "iVBORw0KGgoAAAANS...",
        "confidence_threshold": 0.3,
        "top_k": 3
    }
    
    Example Response:
    {
        "predictions": [
            {"class_name": "cat", "class_id": 3, "confidence": 0.92},
            {"class_name": "dog", "class_id": 5, "confidence": 0.05},
            {"class_name": "bird", "class_id": 2, "confidence": 0.02}
        ],
        "model_name": "resnet18",
        "processing_time": 0.145
    }
    """

@app.post("/predict/batch", response_model=BatchClassificationResponse)
async def classify_batch(request: BatchClassificationRequest):
    """
    Classify multiple images in batch
    
    Features:
    - Process up to 100 images per request
    - Batch inference for efficiency
    - Parallel processing on GPU
    - Individual error handling
    
    Benefits:
    - 3-5x faster than sequential predictions
    - Optimal GPU utilization
    - Amortized preprocessing cost
    
    Use Cases:
    - Bulk image categorization
    - Video frame analysis
    - Dataset annotation
    """

@app.post("/train", response_model=ModelTrainingResponse)
async def train_model(request: ModelTrainingRequest):
    """
    Train new model or fine-tune existing
    
    Parameters:
    - dataset_name: 'cifar10', 'cifar100', or 'custom'
    - model_name: 'cnn_custom', 'resnet18', 'mobilenet'
    - epochs: Training epochs (default: 50)
    - batch_size: Batch size (default: 64)
    - learning_rate: Learning rate (default: 0.001)
    
    Process:
    1. Validate parameters
    2. Load/prepare dataset
    3. Build model architecture
    4. Train with validation
    5. Save best model
    6. Return metrics
    
    Returns: Training metrics and model info
    """
```

---

## Architectural Documentation

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     CLIENT APPLICATIONS                          │
│   (Web Apps, Mobile Apps, Image Processing Pipelines)           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ HTTPS REST API
                            │ POST /predict (base64 image)
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                      FASTAPI APPLICATION                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │           API Endpoints (main.py)                          │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐ │ │
│  │  │  /predict    │  │  /predict/   │  │  /train         │ │ │
│  │  │  (Single)    │  │  batch       │  │  (Model)        │ │ │
│  │  └──────────────┘  └──────────────┘  └─────────────────┘ │ │
│  └────────────────────────────────────────────────────────────┘ │
│                            │                                     │
│  ┌────────────────────────▼────────────────────────────────────┐│
│  │       Image Processing Pipeline                             ││
│  │  ┌───────────────────────────────────────────────────────┐ ││
│  │  │ 1. Base64 Decode                                       │ ││
│  │  │    - Convert string to binary                          │ ││
│  │  │    - Validate image format                             │ ││
│  │  └───────────────────────────────────────────────────────┘ ││
│  │  ┌───────────────────────────────────────────────────────┐ ││
│  │  │ 2. Image Preprocessing                                 │ ││
│  │  │    - Resize to model input size                        │ ││
│  │  │    - Convert color space (RGB)                         │ ││
│  │  │    - Normalize pixel values [0, 1]                     │ ││
│  │  │    - Add batch dimension                               │ ││
│  │  └───────────────────────────────────────────────────────┘ ││
│  │  ┌───────────────────────────────────────────────────────┐ ││
│  │  │ 3. Model Inference                                     │ ││
│  │  │    - Forward pass through CNN                          │ ││
│  │  │    - Get class probabilities (softmax)                 │ ││
│  │  │    - Extract top-K predictions                         │ ││
│  │  └───────────────────────────────────────────────────────┘ ││
│  │  ┌───────────────────────────────────────────────────────┐ ││
│  │  │ 4. Post-processing                                     │ ││
│  │  │    - Apply confidence threshold                        │ ││
│  │  │    - Map class IDs to names                            │ ││
│  │  │    - Format JSON response                              │ ││
│  │  └───────────────────────────────────────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────┘│
│                            │                                     │
│  ┌────────────────────────▼────────────────────────────────────┐│
│  │         Deep Learning Model (TensorFlow/Keras)              ││
│  │                                                              ││
│  │  ┌─────────────────────────────────────────────────────┐   ││
│  │  │  Convolutional Neural Network                       │   ││
│  │  │  ┌──────────────────────────────────────────────┐   │   ││
│  │  │  │  Input Layer (224x224x3)                     │   │   ││
│  │  │  └──────────────┬───────────────────────────────┘   │   ││
│  │  │                 │                                    │   ││
│  │  │  ┌──────────────▼───────────────────────────────┐   │   ││
│  │  │  │  Convolutional Blocks                        │   │   ││
│  │  │  │  - Conv2D + BatchNorm + ReLU                 │   │   ││
│  │  │  │  - MaxPooling                                 │   │   ││
│  │  │  │  - Dropout                                    │   │   ││
│  │  │  │  (Repeated 3-4 times)                         │   │   ││
│  │  │  └──────────────┬───────────────────────────────┘   │   ││
│  │  │                 │                                    │   ││
│  │  │  ┌──────────────▼───────────────────────────────┐   │   ││
│  │  │  │  Feature Extraction                          │   │   ││
│  │  │  │  - Flatten / Global Pooling                  │   │   ││
│  │  │  │  - Dense layers                               │   │   ││
│  │  │  └──────────────┬───────────────────────────────┘   │   ││
│  │  │                 │                                    │   ││
│  │  │  ┌──────────────▼───────────────────────────────┐   │   ││
│  │  │  │  Classification Head                         │   │   ││
│  │  │  │  - Dense(num_classes)                        │   │   ││
│  │  │  │  - Softmax activation                        │   │   ││
│  │  │  │  - Output: Class probabilities               │   │   ││
│  │  │  └──────────────────────────────────────────────┘   │   ││
│  │  └─────────────────────────────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ Load/Save
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    MODEL STORAGE                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │  Model Files     │  │  Training Data   │  │  Metadata     │ │
│  │  - model.h5      │  │  - CIFAR-10      │  │  - classes    │ │
│  │  - weights.h5    │  │  - CIFAR-100     │  │  - metrics    │ │
│  │  - architecture  │  │  - Custom        │  │  - config     │ │
│  └──────────────────┘  └──────────────────┘  └───────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow - Image Classification

```
┌──────────────┐
│  Input Image │
│  (Any size)  │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│  Decode Base64       │
│  → Binary data       │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Load Image          │
│  → PIL Image object  │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Convert to RGB      │
│  (if grayscale/RGBA) │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Resize Image        │
│  → (224, 224, 3)     │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Normalize Pixels    │
│  [0, 255] → [0, 1]   │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Add Batch Dimension │
│  (H,W,C) → (1,H,W,C) │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  CNN Forward Pass    │
│  - Convolution       │
│  - Activation        │
│  - Pooling           │
│  (Multiple layers)   │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Softmax Layer       │
│  → Class probs       │
│  [0.92, 0.05, 0.02]  │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Top-K Selection     │
│  Sort by confidence  │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Apply Threshold     │
│  Filter low conf     │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Format Response     │
│  {class, conf, id}   │
└──────────────────────┘
```

### Training Pipeline

```
START
  │
  ▼
[Load Dataset]
  │ (CIFAR-10, CIFAR-100, custom)
  │
  ▼
[Split Data]
  │ 80% train, 20% validation
  │
  ▼
[Data Augmentation Setup]
  │ - Rotation, flip, zoom, shift
  │
  ▼
[Build Model Architecture]
  │ - Custom CNN / ResNet / MobileNet
  │
  ▼
[Compile Model]
  │ - Optimizer: Adam
  │ - Loss: Categorical crossentropy
  │ - Metrics: Accuracy
  │
  ▼
[Set Up Callbacks]
  │ - Early stopping
  │ - Learning rate reduction
  │ - Model checkpoint
  │
  ▼
[Training Loop]
  │
  ├─ For each epoch:
  │   │
  │   ├─ For each batch:
  │   │   │
  │   │   ├─ Apply augmentation
  │   │   ├─ Forward pass
  │   │   ├─ Calculate loss
  │   │   ├─ Backpropagation
  │   │   └─ Update weights
  │   │
  │   ├─ Validation step
  │   │   │
  │   │   ├─ Forward pass (no augmentation)
  │   │   ├─ Calculate val_loss
  │   │   └─ Calculate val_accuracy
  │   │
  │   ├─ Check early stopping
  │   │   │
  │   │   └─ If no improvement for 10 epochs → STOP
  │   │
  │   └─ Adjust learning rate if needed
  │
  ▼
[Evaluate on Test Set]
  │ - Final accuracy
  │ - Top-5 accuracy
  │ - Confusion matrix
  │
  ▼
[Save Best Model]
  │ - Model architecture
  │ - Trained weights
  │ - Metadata
  │
  ▼
END
```

---

## Project Structure

```
image-classification/
│
├── app/                          # Application source code
│   ├── __init__.py              #