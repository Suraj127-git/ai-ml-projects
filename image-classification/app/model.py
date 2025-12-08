import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import joblib
import base64
import io
from PIL import Image
import time
import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class ImageClassificationModel:
    """CNN-based Image Classification Model"""
    
    def __init__(self):
        self.model = None
        self.model_name = None
        self.dataset_name = None
        self.class_names = []
        self.input_shape = (32, 32, 3)  # Default CIFAR-10 size
        self.num_classes = 10
        self.is_trained = False
        
    def create_cnn_model(self, input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
        """Create a custom CNN architecture"""
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def create_resnet_model(self, input_shape: Tuple[int, int, int], num_classes: int, version: str = "resnet18") -> keras.Model:
        """Create ResNet model (simplified version for demonstration)"""
        
        def residual_block(x, filters, kernel_size=3, stride=1, activation='relu'):
            """Residual block"""
            shortcut = x
            
            # First conv
            x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(activation)(x)
            
            # Second conv
            x = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            
            # Shortcut connection
            if stride != 1 or shortcut.shape[-1] != filters:
                shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
                shortcut = layers.BatchNormalization()(shortcut)
            
            x = layers.Add()([x, shortcut])
            x = layers.Activation(activation)(x)
            
            return x
        
        # Input
        inputs = layers.Input(shape=input_shape)
        x = inputs
        
        # Initial conv
        x = layers.Conv2D(64, 7, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
        
        # Residual blocks
        if version == "resnet18":
            # ResNet-18 architecture
            x = residual_block(x, 64)
            x = residual_block(x, 64)
            x = residual_block(x, 128, stride=2)
            x = residual_block(x, 128)
            x = residual_block(x, 256, stride=2)
            x = residual_block(x, 256)
            x = residual_block(x, 512, stride=2)
            x = residual_block(x, 512)
        else:  # ResNet-50
            # ResNet-50 architecture (simplified)
            x = residual_block(x, 64)
            x = residual_block(x, 64)
            x = residual_block(x, 128, stride=2)
            x = residual_block(x, 128)
            x = residual_block(x, 256, stride=2)
            x = residual_block(x, 256)
            x = residual_block(x, 512, stride=2)
            x = residual_block(x, 512)
        
        # Global average pooling and classifier
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, x, name=version)
        return model
    
    def create_mobilenet_model(self, input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
        """Create MobileNet model (simplified version)"""
        
        def depthwise_separable_conv(x, filters, kernel_size=3, stride=1):
            """Depthwise separable convolution"""
            # Depthwise conv
            x = layers.DepthwiseConv2D(kernel_size, strides=stride, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU(6.)(x)
            
            # Pointwise conv
            x = layers.Conv2D(filters, 1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU(6.)(x)
            
            return x
        
        # Input
        inputs = layers.Input(shape=input_shape)
        x = inputs
        
        # Initial conv
        x = layers.Conv2D(32, 3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.)(x)
        
        # MobileNet blocks
        x = depthwise_separable_conv(x, 64)
        x = depthwise_separable_conv(x, 128, stride=2)
        x = depthwise_separable_conv(x, 128)
        x = depthwise_separable_conv(x, 256, stride=2)
        x = depthwise_separable_conv(x, 256)
        x = depthwise_separable_conv(x, 512, stride=2)
        
        # Additional blocks
        for _ in range(5):
            x = depthwise_separable_conv(x, 512)
        
        x = depthwise_separable_conv(x, 1024, stride=2)
        x = depthwise_separable_conv(x, 1024)
        
        # Global average pooling and classifier
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, x, name='mobilenet')
        return model
    
    def create_model(self, model_name: str, input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
        """Create model based on name"""
        if model_name == "cnn_custom":
            return self.create_cnn_model(input_shape, num_classes)
        elif model_name == "resnet18":
            return self.create_resnet_model(input_shape, num_classes, "resnet18")
        elif model_name == "resnet50":
            return self.create_resnet_model(input_shape, num_classes, "resnet50")
        elif model_name == "mobilenet":
            return self.create_mobilenet_model(input_shape, num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def load_dataset(self, dataset_name: str):
        """Load dataset"""
        if dataset_name == "cifar10":
            (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
            self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                              'dog', 'frog', 'horse', 'ship', 'truck']
            self.num_classes = 10
            self.input_shape = (32, 32, 3)
            
        elif dataset_name == "cifar100":
            (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
            self.class_names = [f"class_{i}" for i in range(100)]
            self.num_classes = 100
            self.input_shape = (32, 32, 3)
            
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        self.dataset_name = dataset_name
        
        # Normalize pixel values
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Convert labels to categorical
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        
        return (x_train, y_train), (x_test, y_test)
    
    def preprocess_image(self, image_base64: str, target_size: Tuple[int, int] = None) -> np.ndarray:
        """Preprocess base64 image for classification"""
        if target_size is None:
            target_size = self.input_shape[:2]
        
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Normalize pixel values
        image_array = image_array.astype('float32') / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    def train_model(self, dataset_name: str, model_name: str, epochs: int = 10, 
                   batch_size: int = 32, learning_rate: float = 0.001, 
                   validation_split: float = 0.2) -> Dict:
        """Train the model"""
        start_time = time.time()
        
        # Load dataset
        (x_train, y_train), (x_test, y_test) = self.load_dataset(dataset_name)
        
        # Create model
        self.model = self.create_model(model_name, self.input_shape, self.num_classes)
        self.model_name = model_name
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            validation_split=validation_split
        )
        
        # Training callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001
        )
        
        # Train model
        history = self.model.fit(
            datagen.flow(x_train, y_train, batch_size=batch_size, subset='training'),
            epochs=epochs,
            validation_data=datagen.flow(x_train, y_train, batch_size=batch_size, subset='validation'),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate on test set
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        
        training_time = time.time() - start_time
        
        # Prepare metrics history
        metrics_history = []
        for i in range(len(history.history['loss'])):
            metrics = {
                'epoch': i + 1,
                'train_loss': float(history.history['loss'][i]),
                'train_accuracy': float(history.history['accuracy'][i]),
                'val_loss': float(history.history['val_loss'][i]),
                'val_accuracy': float(history.history['val_accuracy'][i])
            }
            metrics_history.append(metrics)
        
        self.is_trained = True
        
        return {
            'message': f'Model {model_name} trained successfully on {dataset_name}',
            'model_name': model_name,
            'dataset_name': dataset_name,
            'epochs_completed': len(history.history['loss']),
            'final_train_accuracy': float(history.history['accuracy'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
            'test_accuracy': float(test_accuracy),
            'training_time': training_time,
            'metrics_history': metrics_history
        }
    
    def predict(self, image_base64: str, confidence_threshold: float = 0.5, 
                top_k: int = 5) -> Dict:
        """Make prediction on image"""
        if not self.is_trained:
            raise ValueError("Model is not trained. Please train the model first.")
        
        start_time = time.time()
        
        # Preprocess image
        processed_image = self.preprocess_image(image_base64)
        
        # Get original image dimensions
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        original_width, original_height = image.size
        
        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)
        
        # Get top-k predictions
        top_indices = np.argsort(predictions[0])[-top_k:][::-1]
        
        results = []
        for i, idx in enumerate(top_indices):
            confidence = float(predictions[0][idx])
            if confidence >= confidence_threshold:
                results.append({
                    'class_name': self.class_names[idx] if idx < len(self.class_names) else f'class_{idx}',
                    'class_id': int(idx),
                    'confidence': confidence,
                    'probability': confidence
                })
        
        processing_time = time.time() - start_time
        
        return {
            'predictions': results,
            'model_name': self.model_name,
            'image_width': original_width,
            'image_height': original_height,
            'processing_time': processing_time
        }
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model is not trained. Please train the model first.")
        
        # Save Keras model
        model_path = filepath + '_model.h5'
        self.model.save(model_path)
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'dataset_name': self.dataset_name,
            'class_names': self.class_names,
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'is_trained': self.is_trained,
            'saved_date': datetime.now().isoformat()
        }
        
        metadata_path = filepath + '_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            'model_path': model_path,
            'metadata_path': metadata_path
        }
    
    def load_model(self, filepath: str):
        """Load a saved model"""
        # Load Keras model
        model_path = filepath + '_model.h5'
        self.model = keras.models.load_model(model_path)
        
        # Load metadata
        metadata_path = filepath + '_metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.model_name = metadata['model_name']
        self.dataset_name = metadata['dataset_name']
        self.class_names = metadata['class_names']
        self.input_shape = tuple(metadata['input_shape'])
        self.num_classes = metadata['num_classes']
        self.is_trained = metadata['is_trained']
        
        return metadata
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        if not self.is_trained:
            return {
                'model_name': self.model_name or 'Not loaded',
                'dataset_name': self.dataset_name or 'Not loaded',
                'class_names': self.class_names,
                'input_shape': self.input_shape,
                'num_classes': self.num_classes,
                'is_trained': self.is_trained,
                'total_parameters': 0,
                'model_size_mb': 0
            }
        
        # Count total parameters
        total_params = self.model.count_params()
        
        # Estimate model size (rough approximation)
        model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        return {
            'model_name': self.model_name,
            'dataset_name': self.dataset_name,
            'class_names': self.class_names,
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'is_trained': self.is_trained,
            'total_parameters': total_params,
            'model_size_mb': model_size_mb
        }