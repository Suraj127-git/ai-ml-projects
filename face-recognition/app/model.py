import cv2
import numpy as np
import base64
from PIL import Image
import io
import os
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
import time
import uuid
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FaceRecognitionModel:
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize Face Recognition Model with FaceNet"""
        self.device = device
        self.mtcnn = None
        self.facenet = None
        self.registered_persons = {}
        self.detection_threshold = 0.5
        self.recognition_threshold = 0.6
        self.embedding_dimension = 512
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize MTCNN and FaceNet models"""
        try:
            # Initialize MTCNN for face detection
            self.mtcnn = MTCNN(
                image_size=160,
                margin=0,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=True,
                device=self.device
            )
            
            # Initialize FaceNet for face recognition
            self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            
            print(f"Models initialized successfully on {self.device}")
            
        except Exception as e:
            raise Exception(f"Error initializing models: {str(e)}")
    
    def _base64_to_image(self, base64_string: str) -> np.ndarray:
        """Convert base64 string to numpy array image"""
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_string)
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(image)
            
            return image_array
            
        except Exception as e:
            raise Exception(f"Error converting base64 to image: {str(e)}")
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy array image to base64 string"""
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(image)
            
            # Save to BytesIO
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG')
            buffer.seek(0)
            
            # Convert to base64
            base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return base64_string
            
        except Exception as e:
            raise Exception(f"Error converting image to base64: {str(e)}")
    
    def detect_faces(self, image_base64: str, confidence_threshold: float = 0.5, extract_faces: bool = True) -> Dict:
        """Detect faces in an image"""
        start_time = time.time()
        
        try:
            # Convert base64 to image
            image = self._base64_to_image(image_base64)
            original_height, original_width = image.shape[:2]
            
            # Detect faces using MTCNN
            boxes, probs, landmarks = self.mtcnn.detect(image, landmarks=True)
            
            faces = []
            
            if boxes is not None:
                for i, (box, prob, landmark) in enumerate(zip(boxes, probs, landmarks)):
                    if prob >= confidence_threshold:
                        # Convert box coordinates to integers
                        x1, y1, x2, y2 = [int(coord) for coord in box]
                        
                        # Ensure coordinates are within image bounds
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(original_width, x2)
                        y2 = min(original_height, y2)
                        
                        face_data = {
                            "face_id": str(uuid.uuid4()),
                            "bbox": {
                                "x": x1,
                                "y": y1,
                                "width": x2 - x1,
                                "height": y2 - y1
                            },
                            "confidence": float(prob)
                        }
                        
                        # Extract face image if requested
                        if extract_faces:
                            face_crop = image[y1:y2, x1:x2]
                            if face_crop.size > 0:
                                face_data["face_image_base64"] = self._image_to_base64(face_crop)
                        
                        # Add landmarks if available
                        if landmark is not None:
                            face_data["landmarks"] = {
                                "left_eye": [int(landmark[0][0]), int(landmark[0][1])],
                                "right_eye": [int(landmark[1][0]), int(landmark[1][1])],
                                "nose": [int(landmark[2][0]), int(landmark[2][1])],
                                "mouth_left": [int(landmark[3][0]), int(landmark[3][1])],
                                "mouth_right": [int(landmark[4][0]), int(landmark[4][1])]
                            }
                        
                        faces.append(face_data)
            
            processing_time = time.time() - start_time
            
            return {
                "faces": faces,
                "total_faces": len(faces),
                "image_width": original_width,
                "image_height": original_height,
                "processing_time": processing_time
            }
            
        except Exception as e:
            raise Exception(f"Error detecting faces: {str(e)}")
    
    def extract_face_embedding(self, face_image_base64: str) -> np.ndarray:
        """Extract face embedding from a face image"""
        try:
            # Convert base64 to image
            face_image = self._base64_to_image(face_image_base64)
            
            # Convert to PIL Image
            face_pil = Image.fromarray(face_image)
            
            # Resize to 160x160 (FaceNet input size)
            face_pil = face_pil.resize((160, 160))
            
            # Convert to tensor and add batch dimension
            face_tensor = torch.tensor(np.array(face_pil)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            face_tensor = face_tensor.to(self.device)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.facenet(face_tensor)
            
            # Convert to numpy array
            embedding_array = embedding.cpu().numpy().flatten()
            
            return embedding_array
            
        except Exception as e:
            raise Exception(f"Error extracting face embedding: {str(e)}")
    
    def register_person(self, person_name: str, face_images: List[str]) -> Dict:
        """Register a new person with multiple face images"""
        try:
            person_id = str(uuid.uuid4())
            embeddings = []
            
            # Extract embeddings from all face images
            for face_image in face_images:
                try:
                    embedding = self.extract_face_embedding(face_image)
                    embeddings.append(embedding)
                except Exception as e:
                    print(f"Warning: Could not extract embedding from one image: {str(e)}")
                    continue
            
            if not embeddings:
                raise Exception("No valid face embeddings could be extracted")
            
            # Calculate average embedding
            average_embedding = np.mean(embeddings, axis=0)
            
            # Store person information
            self.registered_persons[person_id] = {
                "person_name": person_name,
                "face_embedding": average_embedding,
                "registration_date": datetime.now().isoformat(),
                "image_count": len(embeddings),
                "embeddings": embeddings
            }
            
            return {
                "person_id": person_id,
                "person_name": person_name,
                "embeddings_created": len(embeddings),
                "registration_status": "success",
                "average_embedding": {
                    "embedding": average_embedding.tolist(),
                    "face_id": f"{person_id}_avg"
                }
            }
            
        except Exception as e:
            raise Exception(f"Error registering person: {str(e)}")
    
    def recognize_face(self, face_image_base64: str, known_faces: List[Dict], similarity_threshold: float = 0.6) -> Dict:
        """Recognize a face against known faces"""
        try:
            # Extract embedding from input face
            input_embedding = self.extract_face_embedding(face_image_base64)
            
            # Calculate similarities with all known faces
            similarities = []
            
            for known_face in known_faces:
                known_embedding = np.array(known_face["embedding"])
                
                # Calculate cosine similarity
                similarity = cosine_similarity([input_embedding], [known_embedding])[0][0]
                
                similarities.append({
                    "face_id": known_face["face_id"],
                    "similarity": float(similarity)
                })
            
            # Find best match
            if similarities:
                best_match = max(similarities, key=lambda x: x["similarity"])
                best_similarity = best_match["similarity"]
                
                # Check if similarity exceeds threshold
                if best_similarity >= similarity_threshold:
                    return {
                        "recognized": True,
                        "matched_face_id": best_match["face_id"],
                        "similarity_score": float(best_similarity),
                        "confidence": float(best_similarity),
                        "all_similarities": similarities
                    }
                else:
                    return {
                        "recognized": False,
                        "matched_face_id": None,
                        "similarity_score": float(best_similarity),
                        "confidence": float(best_similarity),
                        "all_similarities": similarities
                    }
            else:
                return {
                    "recognized": False,
                    "matched_face_id": None,
                    "similarity_score": 0.0,
                    "confidence": 0.0,
                    "all_similarities": []
                }
                
        except Exception as e:
            raise Exception(f"Error recognizing face: {str(e)}")
    
    def compare_faces(self, image1_base64: str, image2_base64: str) -> Dict:
        """Compare two face images to see if they belong to the same person"""
        try:
            # Extract embeddings from both images
            embedding1 = self.extract_face_embedding(image1_base64)
            embedding2 = self.extract_face_embedding(image2_base64)
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            
            # Determine if same person (threshold can be adjusted)
            are_same_person = similarity >= self.recognition_threshold
            
            return {
                "similarity_score": float(similarity),
                "are_same_person": bool(are_same_person),
                "confidence": float(similarity)
            }
            
        except Exception as e:
            raise Exception(f"Error comparing faces: {str(e)}")
    
    def get_registered_persons(self) -> List[Dict]:
        """Get list of all registered persons"""
        persons = []
        
        for person_id, person_data in self.registered_persons.items():
            persons.append({
                "person_id": person_id,
                "person_name": person_data["person_name"],
                "face_embedding": {
                    "embedding": person_data["face_embedding"].tolist(),
                    "face_id": f"{person_id}_avg"
                },
                "registration_date": person_data["registration_date"],
                "image_count": person_data["image_count"]
            })
        
        return persons
    
    def delete_person(self, person_id: str) -> bool:
        """Delete a registered person"""
        if person_id in self.registered_persons:
            del self.registered_persons[person_id]
            return True
        return False
    
    def save_models(self, filepath: str):
        """Save registered persons data"""
        import joblib
        
        # Convert numpy arrays to lists for serialization
        data_to_save = {}
        for person_id, person_data in self.registered_persons.items():
            data_to_save[person_id] = {
                "person_name": person_data["person_name"],
                "face_embedding": person_data["face_embedding"].tolist(),
                "registration_date": person_data["registration_date"],
                "image_count": person_data["image_count"]
            }
        
        joblib.dump(data_to_save, filepath)
    
    def load_models(self, filepath: str):
        """Load registered persons data"""
        import joblib
        
        loaded_data = joblib.load(filepath)
        
        for person_id, person_data in loaded_data.items():
            self.registered_persons[person_id] = {
                "person_name": person_data["person_name"],
                "face_embedding": np.array(person_data["face_embedding"]),
                "registration_date": person_data["registration_date"],
                "image_count": person_data["image_count"],
                "embeddings": []  # Not saved, will be empty after loading
            }
    
    def get_model_info(self) -> Dict:
        """Get information about the face recognition model"""
        return {
            "model_name": "FaceNet",
            "embedding_dimension": self.embedding_dimension,
            "supported_image_formats": ["jpg", "jpeg", "png", "bmp"],
            "max_image_size": 10 * 1024 * 1024,  # 10MB
            "detection_confidence_threshold": self.detection_threshold,
            "recognition_threshold": self.recognition_threshold,
            "device": self.device,
            "registered_persons_count": len(self.registered_persons)
        }