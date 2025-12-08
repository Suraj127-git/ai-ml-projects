import numpy as np
import pandas as pd
import base64
import io
from PIL import Image
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json
import hashlib
import uuid
from pathlib import Path

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib

# Computer Vision imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models, transforms
    from torchvision.models import ResNet50_Weights
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Using fallback methods.")

# Custom imports
from .schemas import (
    QualityControlRequest, QualityControlResponse, DefectDetection, 
    QualityMetrics, QualityStatus, DefectType, ProductCategory,
    BatchQualityRequest, BatchQualityResponse, ModelInfo, QualityStandards
)

class QualityControlCNN(nn.Module):
    """Custom CNN for defect detection"""
    
    def __init__(self, num_defect_types: int = 8, num_categories: int = 8):
        super(QualityControlCNN, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Classification head
        self.fc1 = nn.Linear(512 * 14 * 14, 1024)  # Assuming 224x224 input
        self.fc2 = nn.Linear(1024, 512)
        self.fc_defect = nn.Linear(512, num_defect_types)
        self.fc_quality = nn.Linear(512, 1)
        self.fc_category = nn.Linear(512, num_categories)
        
    def forward(self, x):
        # Feature extraction
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        defect_output = torch.sigmoid(self.fc_defect(x))
        quality_output = torch.sigmoid(self.fc_quality(x))
        category_output = torch.softmax(self.fc_category(x), dim=1)
        
        return defect_output, quality_output.squeeze(), category_output

class ResNetQualityClassifier:
    """ResNet-based quality classifier using transfer learning"""
    
    def __init__(self, num_defect_types: int = 8):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_defect_types = num_defect_types
        
        if TORCH_AVAILABLE:
            # Load pre-trained ResNet50
            self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            
            # Freeze early layers
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Replace final layer
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_defect_types + 1)  # +1 for quality score
            )
            
            self.model.to(self.device)
            
            # Image preprocessing
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            # Fallback to traditional ML
            self.fallback_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.scaler = StandardScaler()
            self.is_trained = False
    
    def extract_features(self, image: Image.Image) -> np.ndarray:
        """Extract features from image for fallback model"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize for consistency
        image = image.resize((128, 128))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Extract basic features
        features = []
        
        # Color statistics
        for channel in range(3):
            channel_data = img_array[:, :, channel]
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.min(channel_data),
                np.max(channel_data)
            ])
        
        # Edge detection (simple gradient)
        gray = np.mean(img_array, axis=2)
        grad_x = np.gradient(gray, axis=1)
        grad_y = np.gradient(gray, axis=0)
        edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features.extend([
            np.mean(edge_magnitude),
            np.std(edge_magnitude),
            np.max(edge_magnitude)
        ])
        
        # Texture features (simple)
        features.extend([
            np.mean(gray),
            np.std(gray),
            np.percentile(gray, 25),
            np.percentile(gray, 75)
        ])
        
        return np.array(features)
    
    def predict(self, image: Image.Image) -> Tuple[List[float], float, Dict]:
        """Predict defects and quality score"""
        if TORCH_AVAILABLE and self.model:
            # Use ResNet model
            self.model.eval()
            
            with torch.no_grad():
                # Preprocess image
                image_tensor = self.transform(image).unsqueeze(0).to(self.device)
                
                # Get predictions
                outputs = self.model(image_tensor)
                
                # Split outputs
                defect_scores = torch.sigmoid(outputs[:, :self.num_defect_types])
                quality_score = torch.sigmoid(outputs[:, self.num_defect_types])
                
                return (
                    defect_scores.cpu().numpy()[0].tolist(),
                    quality_score.cpu().numpy()[0].item(),
                    {"model_type": "resnet", "device": str(self.device)}
                )
        else:
            # Use fallback model
            if not self.is_trained:
                # Return dummy predictions for demo
                defect_scores = np.random.uniform(0, 0.3, self.num_defect_types)
                quality_score = np.random.uniform(0.7, 0.95)
                
                return defect_scores.tolist(), quality_score, {"model_type": "fallback"}
            
            # Extract features and predict
            features = self.extract_features(image)
            features_scaled = self.scaler.transform([features])
            
            # Dummy prediction for now
            defect_scores = np.random.uniform(0, 0.3, self.num_defect_types)
            quality_score = np.random.uniform(0.7, 0.95)
            
            return defect_scores.tolist(), quality_score, {"model_type": "ml_fallback"}

class QualityControlModel:
    """Main quality control model"""
    
    def __init__(self):
        self.classifier = ResNetQualityClassifier()
        self.quality_standards = self._load_quality_standards()
        self.model_version = "1.0.0"
        self.processing_stats = {"total_inspected": 0, "defects_found": 0}
    
    def _load_quality_standards(self) -> Dict[ProductCategory, QualityStandards]:
        """Load quality standards for different product categories"""
        return {
            ProductCategory.ELECTRONICS: QualityStandards(
                category=ProductCategory.ELECTRONICS,
                max_defects=2,
                min_quality_score=0.85,
                critical_defect_threshold=0.8,
                dimensional_tolerance=0.05
            ),
            ProductCategory.AUTOMOTIVE: QualityStandards(
                category=ProductCategory.AUTOMOTIVE,
                max_defects=1,
                min_quality_score=0.90,
                critical_defect_threshold=0.85,
                dimensional_tolerance=0.02
            ),
            ProductCategory.TEXTILES: QualityStandards(
                category=ProductCategory.TEXTILES,
                max_defects=3,
                min_quality_score=0.80,
                critical_defect_threshold=0.75,
                color_variance_threshold=0.1
            ),
            ProductCategory.FOOD: QualityStandards(
                category=ProductCategory.FOOD,
                max_defects=1,
                min_quality_score=0.92,
                critical_defect_threshold=0.90,
                color_variance_threshold=0.05
            ),
            ProductCategory.PHARMACEUTICAL: QualityStandards(
                category=ProductCategory.PHARMACEUTICAL,
                max_defects=0,
                min_quality_score=0.95,
                critical_defect_threshold=0.95,
                dimensional_tolerance=0.01
            ),
            ProductCategory.METAL: QualityStandards(
                category=ProductCategory.METAL,
                max_defects=2,
                min_quality_score=0.88,
                critical_defect_threshold=0.80,
                dimensional_tolerance=0.03
            ),
            ProductCategory.PLASTIC: QualityStandards(
                category=ProductCategory.PLASTIC,
                max_defects=3,
                min_quality_score=0.82,
                critical_defect_threshold=0.75,
                dimensional_tolerance=0.08
            ),
            ProductCategory.CERAMIC: QualityStandards(
                category=ProductCategory.CERAMIC,
                max_defects=2,
                min_quality_score=0.85,
                critical_defect_threshold=0.80,
                color_variance_threshold=0.08
            )
        }
    
    def decode_image(self, image_data: str, image_format: str) -> Image.Image:
        """Decode base64 image data"""
        try:
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            
            # Open image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
        except Exception as e:
            raise ValueError(f"Failed to decode image: {str(e)}")
    
    def detect_defects(self, image: Image.Image, category: ProductCategory) -> List[DefectDetection]:
        """Detect defects in the product image"""
        # Get predictions from classifier
        defect_scores, quality_score, model_info = self.classifier.predict(image)
        
        # Convert scores to defect detections
        defects = []
        defect_types = list(DefectType)
        
        for i, score in enumerate(defect_scores):
            if score > 0.3:  # Detection threshold
                # Generate random location for demo (in real implementation, this would be actual coordinates)
                width, height = image.size
                
                defect = DefectDetection(
                    defect_type=defect_types[i],
                    confidence=float(score),
                    location={
                        "x": np.random.randint(0, max(1, width - 50)),
                        "y": np.random.randint(0, max(1, height - 50)),
                        "width": np.random.randint(20, 100),
                        "height": np.random.randint(20, 100)
                    },
                    severity=float(score * 0.8 + np.random.uniform(0, 0.2)),
                    description=f"{defect_types[i].value.replace('_', ' ').title()} detected in {category.value} product"
                )
                defects.append(defect)
        
        return defects, quality_score
    
    def assess_quality(self, request: QualityControlRequest) -> QualityControlResponse:
        """Perform quality control inspection on a single product"""
        start_time = datetime.now()
        
        try:
            # Decode image
            image = self.decode_image(request.image_data, request.image_format)
            
            # Detect defects
            defects, quality_score = self.detect_defects(image, request.category)
            
            # Get quality standards
            standards = self.quality_standards.get(request.category, self.quality_standards[ProductCategory.ELECTRONICS])
            
            # Determine status
            critical_defects = sum(1 for d in defects if d.confidence > standards.critical_defect_threshold)
            
            if critical_defects > 0 or quality_score < standards.min_quality_score:
                status = QualityStatus.FAIL
            elif len(defects) > standards.max_defects or quality_score < 0.9:
                status = QualityStatus.WARNING
            elif len(defects) > 0:
                status = QualityStatus.NEEDS_REVIEW
            else:
                status = QualityStatus.PASS
            
            # Generate recommendations
            recommendations = self._generate_recommendations(defects, request.category, quality_score)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update stats
            self.processing_stats["total_inspected"] += 1
            self.processing_stats["defects_found"] += len(defects)
            
            # Create quality metrics
            metrics = QualityMetrics(
                overall_score=quality_score,
                defect_count=len(defects),
                critical_defects=critical_defects,
                minor_defects=len(defects) - critical_defects,
                dimensional_accuracy=np.random.uniform(0.85, 0.99),  # Placeholder
                surface_quality=np.random.uniform(0.80, 0.98),      # Placeholder
                color_consistency=np.random.uniform(0.75, 0.95)       # Placeholder
            )
            
            return QualityControlResponse(
                product_id=request.product_id,
                inspection_id=str(uuid.uuid4()),
                status=status,
                quality_score=quality_score,
                defects_detected=defects,
                metrics=metrics,
                processing_time=processing_time,
                recommendations=recommendations,
                inspection_timestamp=datetime.now(),
                model_version=self.model_version
            )
            
        except Exception as e:
            raise RuntimeError(f"Quality control inspection failed: {str(e)}")
    
    def _generate_recommendations(self, defects: List[DefectDetection], 
                                category: ProductCategory, quality_score: float) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        if quality_score < 0.8:
            recommendations.append("Consider improving manufacturing process for better quality")
        
        if len(defects) > 0:
            defect_types = [d.defect_type for d in defects]
            
            if DefectType.SCRATCH in defect_types:
                recommendations.append("Implement better handling procedures to prevent scratches")
            
            if DefectType.CRACK in defect_types:
                recommendations.append("Review material quality and stress testing procedures")
            
            if DefectType.DISCOLORATION in defect_types:
                recommendations.append("Check storage conditions and material compatibility")
            
            if DefectType.DIMENSIONAL_ERROR in defect_types:
                recommendations.append("Calibrate manufacturing equipment and review tolerances")
        
        if not recommendations and quality_score > 0.9:
            recommendations.append("Product quality is excellent - maintain current processes")
        
        return recommendations
    
    def batch_inspect(self, request: BatchQualityRequest) -> BatchQualityResponse:
        """Perform quality control inspection on multiple products"""
        start_time = datetime.now()
        
        individual_results = []
        passed_count = 0
        failed_count = 0
        warning_count = 0
        
        for product_request in request.products:
            try:
                result = self.assess_quality(product_request)
                individual_results.append(result)
                
                if result.status == QualityStatus.PASS:
                    passed_count += 1
                elif result.status == QualityStatus.FAIL:
                    failed_count += 1
                elif result.status == QualityStatus.WARNING:
                    warning_count += 1
                
            except Exception as e:
                # Create failed result for products that couldn't be processed
                failed_result = QualityControlResponse(
                    product_id=product_request.product_id,
                    inspection_id=str(uuid.uuid4()),
                    status=QualityStatus.FAIL,
                    quality_score=0.0,
                    defects_detected=[],
                    metrics=QualityMetrics(
                        overall_score=0.0,
                        defect_count=0,
                        critical_defects=0,
                        minor_defects=0
                    ),
                    processing_time=0.0,
                    recommendations=[f"Inspection failed: {str(e)}"],
                    inspection_timestamp=datetime.now(),
                    model_version=self.model_version
                )
                individual_results.append(failed_result)
                failed_count += 1
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Generate batch summary
        batch_summary = {
            "defect_rate": failed_count / len(request.products) if request.products else 0,
            "warning_rate": warning_count / len(request.products) if request.products else 0,
            "average_quality_score": np.mean([r.quality_score for r in individual_results]) if individual_results else 0,
            "total_defects": sum(len(r.defects_detected) for r in individual_results),
            "most_common_defects": self._get_most_common_defects(individual_results)
        }
        
        return BatchQualityResponse(
            batch_id=request.products[0].batch_id if request.products else str(uuid.uuid4()),
            total_products=len(request.products),
            passed_products=passed_count,
            failed_products=failed_count,
            warning_products=warning_count,
            processing_time=processing_time,
            individual_results=individual_results,
            batch_summary=batch_summary
        )
    
    def _get_most_common_defects(self, results: List[QualityControlResponse]) -> Dict[str, int]:
        """Get most common defects across all results"""
        defect_counts = {}
        
        for result in results:
            for defect in result.defects_detected:
                defect_type = defect.defect_type.value
                defect_counts[defect_type] = defect_counts.get(defect_type, 0) + 1
        
        # Sort by frequency
        return dict(sorted(defect_counts.items(), key=lambda x: x[1], reverse=True)[:5])
    
    def get_model_info(self) -> ModelInfo:
        """Get model information"""
        return ModelInfo(
            model_name="Quality Control CNN/ResNet",
            version=self.model_version,
            accuracy=0.92,  # Placeholder - would be actual validation accuracy
            training_date=datetime.now(),
            defect_types=list(DefectType),
            categories=list(ProductCategory)
        )
    
    def generate_synthetic_training_data(self, n_samples: int = 100) -> List[Dict]:
        """Generate synthetic training data for model development"""
        training_data = []
        
        for i in range(n_samples):
            # Generate random product data
            category = np.random.choice(list(ProductCategory))
            
            # Generate synthetic defect data
            n_defects = np.random.randint(0, 4)
            defects = []
            
            for j in range(n_defects):
                defect = DefectDetection(
                    defect_type=np.random.choice(list(DefectType)),
                    confidence=np.random.uniform(0.3, 0.95),
                    location={
                        "x": np.random.randint(0, 200),
                        "y": np.random.randint(0, 200),
                        "width": np.random.randint(20, 100),
                        "height": np.random.randint(20, 100)
                    },
                    severity=np.random.uniform(0.3, 0.9),
                    description=f"Synthetic defect {j+1}"
                )
                defects.append(defect)
            
            # Calculate quality score based on defects
            quality_score = np.random.uniform(0.6, 0.98)
            if n_defects > 0:
                quality_score *= (1 - 0.1 * n_defects)
            
            training_data.append({
                "product_id": f"synthetic_{i+1}",
                "category": category,
                "defects": defects,
                "quality_score": quality_score,
                "defect_count": n_defects
            })
        
        return training_data