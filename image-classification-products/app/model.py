import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from typing import List, Dict, Optional, Union
import io
import base64
from datetime import datetime
import joblib
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

from .schemas import ProductCategory, ModelType, ImageSize

class SimpleCNN(nn.Module):
    """Simple CNN for product classification"""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Calculate the size after convolutions and pooling
        self.fc1 = nn.Linear(128 * 28 * 28, 512)  # Assuming 224x224 input
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class ImageClassificationModel:
    """Image Classification Model for Products"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_type = None
        self.categories = list(ProductCategory)
        self.category_to_idx = {cat.value: i for i, cat in enumerate(self.categories)}
        self.idx_to_category = {i: cat.value for i, cat in enumerate(self.categories)}
        
        # Image preprocessing transforms
        self.transform_small = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.transform_medium = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.transform_large = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_type: ModelType = ModelType.EFFICIENTNET_B0):
        """Load or create a classification model"""
        try:
            if model_type == ModelType.EFFICIENTNET_B0:
                self.model = models.efficientnet_b0(pretrained=True)
                num_features = self.model.classifier[1].in_features
                self.model.classifier[1] = nn.Linear(num_features, len(self.categories))
            
            elif model_type == ModelType.RESNET50:
                self.model = models.resnet50(pretrained=True)
                num_features = self.model.fc.in_features
                self.model.fc = nn.Linear(num_features, len(self.categories))
            
            elif model_type == ModelType.MOBILENET_V2:
                self.model = models.mobilenet_v2(pretrained=True)
                num_features = self.model.classifier[1].in_features
                self.model.classifier[1] = nn.Linear(num_features, len(self.categories))
            
            elif model_type == ModelType.CUSTOM_CNN:
                self.model = SimpleCNN(num_classes=len(self.categories))
            
            else:
                self.model = models.efficientnet_b0(pretrained=True)
                num_features = self.model.classifier[1].in_features
                self.model.classifier[1] = nn.Linear(num_features, len(self.categories))
            
            self.model.to(self.device)
            self.model_type = model_type
            self.model.eval()
            
            print(f"Model {model_type.value} loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to simple CNN
            self.model = SimpleCNN(num_classes=len(self.categories))
            self.model.to(self.device)
            self.model_type = ModelType.CUSTOM_CNN
            self.model.eval()
    
    def preprocess_image(self, image: Image.Image, image_size: ImageSize = ImageSize.SMALL) -> torch.Tensor:
        """Preprocess image for model input"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply appropriate transform based on size
        if image_size == ImageSize.SMALL:
            return self.transform_small(image).unsqueeze(0)
        elif image_size == ImageSize.MEDIUM:
            return self.transform_medium(image).unsqueeze(0)
        else:  # LARGE
            return self.transform_large(image).unsqueeze(0)
    
    def classify_image(self, image: Image.Image, 
                      model_type: ModelType = ModelType.EFFICIENTNET_B0,
                      top_k: int = 5, 
                      confidence_threshold: float = 0.1,
                      image_size: ImageSize = ImageSize.SMALL) -> List[Dict[str, Union[str, float]]]:
        """Classify a single product image"""
        
        # Load model if different from current
        if self.model is None or self.model_type != model_type:
            self.load_model(model_type)
        
        # Preprocess image
        input_tensor = self.preprocess_image(image, image_size)
        input_tensor = input_tensor.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get top k predictions
            top_probs, top_indices = torch.topk(probabilities[0], min(top_k, len(self.categories)))
            
            predictions = []
            for i in range(len(top_probs)):
                prob = top_probs[i].item()
                if prob >= confidence_threshold:
                    idx = top_indices[i].item()
                    category = self.idx_to_category[idx]
                    predictions.append({
                        "category": category,
                        "confidence": prob,
                        "label": category.replace('_', ' ').title()
                    })
            
            return predictions
        
        return []
    
    def generate_synthetic_product_data(self, n_samples: int = 100) -> List[Dict]:
        """Generate synthetic product image data for testing"""
        synthetic_data = []
        
        for i in range(n_samples):
            # Random category
            category = np.random.choice(self.categories)
            
            # Generate random image
            if category == ProductCategory.ELECTRONICS:
                # Simulate electronics image - darker, more structured
                image_array = np.random.randint(50, 150, (224, 224, 3), dtype=np.uint8)
                # Add some geometric patterns
                for _ in range(5):
                    x, y = np.random.randint(0, 200, 2)
                    w, h = np.random.randint(20, 50, 2)
                    image_array[y:y+h, x:x+w] = np.random.randint(100, 255, (h, w, 3))
            
            elif category == ProductCategory.CLOTHING:
                # Simulate clothing image - softer colors, fabric texture
                base_color = np.random.randint(100, 200, 3)
                image_array = np.full((224, 224, 3), base_color, dtype=np.uint8)
                # Add fabric texture
                noise = np.random.randint(-30, 30, (224, 224, 3))
                image_array = np.clip(image_array + noise, 0, 255).astype(np.uint8)
            
            elif category == ProductCategory.FOOD:
                # Simulate food image - warmer colors
                base_color = np.random.randint([150, 100, 50], [255, 200, 150])
                image_array = np.full((224, 224, 3), base_color, dtype=np.uint8)
                # Add some variation using NumPy instead of cv2
                for _ in range(3):
                    x, y = np.random.randint(0, 200, 2)
                    radius = np.random.randint(10, 30)
                    # Create circular pattern using NumPy
                    y_grid, x_grid = np.ogrid[:224, :224]
                    mask = (x_grid - x)**2 + (y_grid - y)**2 <= radius**2
                    image_array[mask] = np.random.randint(100, 255, 3)
            
            else:
                # Default random image
                image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # Convert to PIL Image
            image = Image.fromarray(image_array)
            
            # Convert to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            synthetic_data.append({
                "image_data": img_base64,
                "category": category.value,
                "label": category.value.replace('_', ' ').title()
            })
        
        return synthetic_data
    
    def train_model(self, training_data: List[Dict], 
                   model_type: ModelType = ModelType.CUSTOM_CNN,
                   epochs: int = 10,
                   learning_rate: float = 0.001,
                   validation_split: float = 0.2) -> Dict[str, float]:
        """Train or fine-tune the classification model"""
        
        try:
            # Load appropriate model
            self.load_model(model_type)
            
            # Prepare training data
            X = []
            y = []
            
            for item in training_data:
                image = item["image"]
                label = item["label"]
                category = item["category"]
                
                # Preprocess image
                processed_image = self.preprocess_image(image)
                X.append(processed_image.squeeze(0))
                y.append(self.category_to_idx[category])
            
            X = torch.stack(X)
            y = torch.tensor(y)
            
            # Split into train and validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )
            
            # Create data loaders
            train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
            val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
            
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            
            train_losses = []
            val_accuracies = []
            
            # Training loop
            for epoch in range(epochs):
                self.model.train()
                train_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                self.model.eval()
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs = self.model(batch_X)
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_y.size(0)
                        val_correct += (predicted == batch_y).sum().item()
                
                val_accuracy = val_correct / val_total
                train_losses.append(train_loss / len(train_loader))
                val_accuracies.append(val_accuracy)
                
                print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}, '
                      f'Val Accuracy: {val_accuracy:.4f}')
            
            # Final evaluation
            final_accuracy = val_accuracies[-1] if val_accuracies else 0.0
            final_loss = train_losses[-1] if train_losses else 0.0
            
            return {
                "epochs_completed": epochs,
                "final_accuracy": final_accuracy,
                "final_loss": final_loss
            }
            
        except Exception as e:
            print(f"Training error: {e}")
            return {
                "epochs_completed": 0,
                "final_accuracy": 0.0,
                "final_loss": 0.0
            }
    
    def evaluate_model(self, test_data: List[Dict], 
                      model_type: ModelType = ModelType.EFFICIENTNET_B0) -> Dict[str, Union[float, List]]:
        """Evaluate model performance on test data"""
        
        try:
            # Ensure model is loaded
            if self.model is None or self.model_type != model_type:
                self.load_model(model_type)
            
            y_true = []
            y_pred = []
            
            self.model.eval()
            
            with torch.no_grad():
                for item in test_data:
                    image = item["image"]
                    true_category = item["true_category"]
                    
                    # Preprocess image
                    input_tensor = self.preprocess_image(image)
                    input_tensor = input_tensor.to(self.device)
                    
                    # Make prediction
                    outputs = self.model(input_tensor)
                    _, predicted = torch.max(outputs, 1)
                    
                    y_true.append(self.category_to_idx[true_category])
                    y_pred.append(predicted.item())
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "confusion_matrix": cm.tolist()
            }
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "confusion_matrix": []
            }
    
    def get_model_info(self) -> List[Dict]:
        """Get information about available models"""
        model_info = []
        
        for model_type in ModelType:
            info = {
                "model_type": model_type.value,
                "description": self._get_model_description(model_type),
                "parameters": self._get_model_parameters(model_type),
                "input_size": self._get_model_input_size(model_type),
                "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            model_info.append(info)
        
        return model_info
    
    def _get_model_description(self, model_type: ModelType) -> str:
        """Get model description"""
        descriptions = {
            ModelType.EFFICIENTNET_B0: "EfficientNet-B0: Lightweight and efficient CNN",
            ModelType.RESNET50: "ResNet-50: Deep residual network with 50 layers",
            ModelType.MOBILENET_V2: "MobileNet-V2: Optimized for mobile devices",
            ModelType.CUSTOM_CNN: "Custom CNN: Simple convolutional neural network"
        }
        return descriptions.get(model_type, "Unknown model type")
    
    def _get_model_parameters(self, model_type: ModelType) -> int:
        """Get approximate model parameters"""
        parameters = {
            ModelType.EFFICIENTNET_B0: 5288548,
            ModelType.RESNET50: 25557032,
            ModelType.MOBILENET_V2: 3504872,
            ModelType.CUSTOM_CNN: 1000000
        }
        return parameters.get(model_type, 0)
    
    def _get_model_input_size(self, model_type: ModelType) -> str:
        """Get model input size"""
        sizes = {
            ModelType.EFFICIENTNET_B0: "224x224",
            ModelType.RESNET50: "224x224",
            ModelType.MOBILENET_V2: "224x224",
            ModelType.CUSTOM_CNN: "224x224"
        }
        return sizes.get(model_type, "224x224")