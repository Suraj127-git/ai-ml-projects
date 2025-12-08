from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
import uvicorn
import base64
import os
from datetime import datetime

from app.schemas import (
    FaceDetectionRequest, FaceDetectionResponse, DetectedFace,
    FaceRecognitionRequest, FaceRecognitionResponse, FaceEmbedding,
    FaceComparisonRequest, FaceComparisonResponse,
    FaceRegistrationRequest, FaceRegistrationResponse, PersonInfo, ModelInfo, HealthCheckResponse
)
from app.model import FaceRecognitionModel

# Initialize FastAPI app
app = FastAPI(
    title="Face Recognition API",
    description="API for face detection, recognition, and comparison using CNN and FaceNet embeddings",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the face recognition model
face_model = FaceRecognitionModel()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Face Recognition API",
        "version": "1.0.0",
        "features": [
            "Face Detection",
            "Face Recognition",
            "Face Comparison",
            "Person Registration"
        ],
        "endpoints": [
            "/detect",
            "/recognize",
            "/compare",
            "/register",
            "/persons",
            "/model/info",
            "/health"
        ]
    }

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        registered_persons=len(face_model.registered_persons),
        models_loaded=True
    )

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the face recognition model"""
    try:
        model_info = face_model.get_model_info()
        return ModelInfo(**model_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@app.post("/detect", response_model=FaceDetectionResponse)
async def detect_faces(request: FaceDetectionRequest):
    """Detect faces in an image"""
    try:
        result = face_model.detect_faces(
            request.image_base64,
            confidence_threshold=request.confidence_threshold,
            extract_faces=request.extract_faces
        )
        
        # Convert faces to DetectedFace objects
        detected_faces = []
        for face_data in result["faces"]:
            detected_faces.append(DetectedFace(**face_data))
        
        return FaceDetectionResponse(
            faces=detected_faces,
            total_faces=result["total_faces"],
            image_width=result["image_width"],
            image_height=result["image_height"],
            processing_time=result["processing_time"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face detection error: {str(e)}")

@app.post("/detect/file", response_model=FaceDetectionResponse)
async def detect_faces_from_file(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5),
    extract_faces: bool = Form(True)
):
    """Detect faces from uploaded image file"""
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Convert to base64
        base64_string = base64.b64encode(contents).decode('utf-8')
        
        # Detect faces
        result = face_model.detect_faces(
            base64_string,
            confidence_threshold=confidence_threshold,
            extract_faces=extract_faces
        )
        
        # Convert faces to DetectedFace objects
        detected_faces = []
        for face_data in result["faces"]:
            detected_faces.append(DetectedFace(**face_data))
        
        return FaceDetectionResponse(
            faces=detected_faces,
            total_faces=result["total_faces"],
            image_width=result["image_width"],
            image_height=result["image_height"],
            processing_time=result["processing_time"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face detection error: {str(e)}")

@app.post("/recognize", response_model=FaceRecognitionResponse)
async def recognize_face(request: FaceRecognitionRequest):
    """Recognize a face against known faces"""
    try:
        # Convert FaceEmbedding objects to dictionaries
        known_faces = [
            {
                "face_id": embedding.face_id,
                "embedding": embedding.embedding
            }
            for embedding in request.known_faces
        ]
        
        result = face_model.recognize_face(
            request.image_base64,
            known_faces,
            similarity_threshold=request.similarity_threshold
        )
        
        return FaceRecognitionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face recognition error: {str(e)}")

@app.post("/compare", response_model=FaceComparisonResponse)
async def compare_faces(request: FaceComparisonRequest):
    """Compare two face images"""
    try:
        result = face_model.compare_faces(
            request.image1_base64,
            request.image2_base64
        )
        
        return FaceComparisonResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face comparison error: {str(e)}")

@app.post("/register", response_model=FaceRegistrationResponse)
async def register_person(request: FaceRegistrationRequest):
    """Register a new person with face images"""
    try:
        result = face_model.register_person(
            request.person_name,
            request.face_images
        )
        
        return FaceRegistrationResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Person registration error: {str(e)}")

@app.get("/persons", response_model=List[PersonInfo])
async def get_registered_persons():
    """Get list of all registered persons"""
    try:
        persons_data = face_model.get_registered_persons()
        
        persons = []
        for person_data in persons_data:
            persons.append(PersonInfo(**person_data))
        
        return persons
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting registered persons: {str(e)}")

@app.get("/persons/{person_id}", response_model=PersonInfo)
async def get_person(person_id: str):
    """Get information about a specific person"""
    try:
        if person_id not in face_model.registered_persons:
            raise HTTPException(status_code=404, detail=f'Person with ID {person_id} not found')
        
        person_data = face_model.registered_persons[person_id]
        
        return PersonInfo(
            person_id=person_id,
            person_name=person_data["person_name"],
            face_embedding=FaceEmbedding(
                embedding=person_data["face_embedding"].tolist(),
                face_id=f"{person_id}_avg"
            ),
            registration_date=person_data["registration_date"],
            image_count=person_data["image_count"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting person: {str(e)}")

@app.delete("/persons/{person_id}")
async def delete_person(person_id: str):
    """Delete a registered person"""
    try:
        success = face_model.delete_person(person_id)
        
        if success:
            return {"message": f"Person with ID {person_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f'Person with ID {person_id} not found')
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting person: {str(e)}")

@app.post("/models/save")
async def save_models(filepath: str = "face_recognition_models.joblib"):
    """Save registered persons data"""
    try:
        face_model.save_models(filepath)
        return {"message": f"Models saved successfully to {filepath}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving models: {str(e)}")

@app.post("/models/load")
async def load_models(filepath: str = "face_recognition_models.joblib"):
    """Load registered persons data"""
    try:
        face_model.load_models(filepath)
        return {"message": f"Models loaded successfully from {filepath}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")

@app.post("/recognize/database")
async def recognize_face_from_database(
    image_base64: str,
    similarity_threshold: float = 0.6
):
    """Recognize a face against all registered persons in the database"""
    try:
        # Get all registered persons
        registered_persons = face_model.get_registered_persons()
        
        if not registered_persons:
            return FaceRecognitionResponse(
                recognized=False,
                matched_face_id=None,
                similarity_score=0.0,
                confidence=0.0,
                all_similarities=[]
            )
        
        # Convert to known faces format
        known_faces = [
            {
                "face_id": person["face_embedding"]["face_id"],
                "embedding": person["face_embedding"]["embedding"]
            }
            for person in registered_persons
        ]
        
        # Recognize face
        result = face_model.recognize_face(
            image_base64,
            known_faces,
            similarity_threshold=similarity_threshold
        )
        
        return FaceRecognitionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face recognition error: {str(e)}")

@app.post("/detect/verify")
async def detect_and_verify_faces(
    image_base64: str,
    person_name: str,
    similarity_threshold: float = 0.6
):
    """Detect faces in an image and verify if any match a specific person"""
    try:
        # First detect faces
        detection_result = face_model.detect_faces(image_base64, extract_faces=True)
        
        # Find the person in registered persons
        target_person_id = None
        for person_id, person_data in face_model.registered_persons.items():
            if person_data["person_name"].lower() == person_name.lower():
                target_person_id = person_id
                break
        
        if not target_person_id:
            return {
                "faces_detected": detection_result["faces"],
                "total_faces": detection_result["total_faces"],
                "person_found": False,
                "message": f"Person '{person_name}' not found in database"
            }
        
        # Create known face for the target person
        target_person = face_model.registered_persons[target_person_id]
        known_faces = [{
            "face_id": f"{target_person_id}_avg",
            "embedding": target_person["face_embedding"].tolist()
        }]
        
        # Check each detected face
        verification_results = []
        for face_data in detection_result["faces"]:
            if "face_image_base64" in face_data:
                recognition_result = face_model.recognize_face(
                    face_data["face_image_base64"],
                    known_faces,
                    similarity_threshold=similarity_threshold
                )
                
                verification_results.append({
                    "face_id": face_data["face_id"],
                    "bbox": face_data["bbox"],
                    "is_target_person": recognition_result["recognized"],
                    "similarity_score": recognition_result["similarity_score"],
                    "confidence": recognition_result["confidence"]
                })
        
        return {
            "faces_detected": detection_result["faces"],
            "total_faces": detection_result["total_faces"],
            "person_found": True,
            "verification_results": verification_results,
            "target_person_matches": sum(1 for result in verification_results if result["is_target_person"])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face detection and verification error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8006)