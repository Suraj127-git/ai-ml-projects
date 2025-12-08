from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import logging
from datetime import datetime
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .model import SpeechToTextModel
from .schemas import (
    TranscriptionRequest, TranscriptionResponse, AudioFormat, ModelName,
    BatchTranscriptionRequest, BatchTranscriptionResponse,
    HealthResponse, ModelInfoResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Speech-to-Text Microservice",
    description="Advanced speech recognition API supporting Wav2Vec2 and Whisper models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
speech_model = SpeechToTextModel()

# Thread pool for handling concurrent requests
executor = ThreadPoolExecutor(max_workers=4)

# In-memory storage for batch job results (in production, use Redis or database)
batch_jobs = {}


@app.on_event("startup")
async def startup_event():
    """Initialize the service on startup."""
    logger.info("Speech-to-Text Microservice starting up...")
    
    # Pre-load default model (Wav2Vec2 base)
    try:
        logger.info("Pre-loading default model: wav2vec2-base")
        speech_model.load_model(ModelName.WAV2VEC2_BASE)
        logger.info("Default model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to pre-load default model: {str(e)}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Speech-to-Text Microservice shutting down...")
    executor.shutdown(wait=True)


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Speech-to-Text Microservice",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "models": ["wav2vec2-base", "wav2vec2-large", "whisper-base", "whisper-small", "whisper-medium"],
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        models_loaded=list(speech_model.models.keys()),
        device=str(speech_model.device)
    )


@app.get("/models", response_model=List[ModelInfoResponse])
async def list_models():
    """List available models and their status."""
    models_info = []
    
    for model_name in ModelName:
        info = speech_model.get_model_info(model_name)
        models_info.append(ModelInfoResponse(
            model_name=model_name.value,
            loaded=info["loaded"],
            architecture=info.get("architecture", "Unknown"),
            supported_languages=info.get("supported_languages", []),
            device=info.get("device", "unknown")
        ))
    
    return models_info


@app.post("/models/{model_name}/load", response_model=Dict[str, str])
async def load_model(model_name: ModelName):
    """Load a specific model."""
    try:
        success = speech_model.load_model(model_name)
        if success:
            return {
                "status": "success",
                "message": f"Model {model_name.value} loaded successfully",
                "model_name": model_name.value
            }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to load model {model_name.value}")
            
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(request: TranscriptionRequest):
    """
    Transcribe audio file to text.
    
    Supports multiple audio formats (WAV, MP3, FLAC, M4A, OGG) and models (Wav2Vec2, Whisper).
    """
    try:
        logger.info(f"Transcription request received - Model: {request.model_name}, Format: {request.audio_format}")
        
        # Perform transcription
        result = speech_model.transcribe_audio(
            audio_data=request.audio_base64,
            audio_format=request.audio_format,
            model_name=request.model_name,
            language=request.language,
            return_timestamps=request.return_timestamps
        )
        
        # Create response
        response = TranscriptionResponse(
            text=result["text"],
            model_name=result["model_name"],
            language=result["language"],
            audio_duration=result["audio_duration"],
            processing_time=result["processing_time"],
            confidence=result.get("confidence"),
            word_count=result["word_count"],
            timestamp=result["timestamp"]
        )
        
        # Add segments if available
        if "segments" in result and result["segments"]:
            response.segments = [TranscriptionSegment(**seg) for seg in result["segments"]]
        
        logger.info(f"Transcription completed successfully - Text length: {len(result['text'])} chars")
        return response
        
    except ValueError as e:
        logger.error(f"Transcription validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/transcribe/batch", response_model=Dict[str, str])
async def transcribe_batch(
    request: BatchTranscriptionRequest,
    background_tasks: BackgroundTasks
):
    """
    Submit a batch transcription job.
    
    Returns a job ID that can be used to check the status and retrieve results.
    """
    try:
        job_id = str(uuid.uuid4())
        
        # Initialize job status
        batch_jobs[job_id] = {
            "status": "processing",
            "created_at": datetime.now(),
            "total_files": len(request.audio_files),
            "completed_files": 0,
            "results": [],
            "errors": []
        }
        
        logger.info(f"Batch transcription job {job_id} started with {len(request.audio_files)} files")
        
        # Process in background
        background_tasks.add_task(
            process_batch_transcription,
            job_id,
            request.audio_files,
            request.model_name,
            request.language
        )
        
        return {
            "job_id": job_id,
            "status": "processing",
            "message": "Batch transcription job submitted successfully",
            "total_files": len(request.audio_files),
            "check_status_at": f"/transcribe/batch/{job_id}/status"
        }
        
    except Exception as e:
        logger.error(f"Batch transcription submission error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch submission failed: {str(e)}")


@app.get("/transcribe/batch/{job_id}/status", response_model=Dict[str, any])
async def get_batch_status(job_id: str):
    """Check the status of a batch transcription job."""
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = batch_jobs[job_id]
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "created_at": job["created_at"],
        "total_files": job["total_files"],
        "completed_files": job["completed_files"],
        "progress_percentage": (job["completed_files"] / job["total_files"]) * 100,
        "results_available": job["status"] == "completed"
    }


@app.get("/transcribe/batch/{job_id}/results", response_model=BatchTranscriptionResponse)
async def get_batch_results(job_id: str):
    """Get the results of a completed batch transcription job."""
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = batch_jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    return BatchTranscriptionResponse(
        job_id=job_id,
        status="completed",
        created_at=job["created_at"],
        completed_at=datetime.now(),
        total_files=job["total_files"],
        successful_files=job["completed_files"] - len(job["errors"]),
        failed_files=len(job["errors"]),
        results=job["results"],
        errors=job["errors"]
    )


@app.post("/transcribe/url", response_model=TranscriptionResponse)
async def transcribe_from_url(
    audio_url: str,
    model_name: ModelName = ModelName.WAV2VEC2_BASE,
    language: Optional[str] = None,
    return_timestamps: bool = False
):
    """
    Transcribe audio from a URL.
    
    Note: This endpoint downloads the audio file from the provided URL.
    """
    try:
        import httpx
        
        logger.info(f"Downloading audio from URL: {audio_url}")
        
        # Download audio file
        async with httpx.AsyncClient() as client:
            response = await client.get(audio_url, timeout=30.0)
            response.raise_for_status()
            
            audio_data = base64.b64encode(response.content).decode('utf-8')
            
            # Try to determine audio format from URL or content type
            content_type = response.headers.get('content-type', '')
            audio_format = infer_audio_format(audio_url, content_type)
        
        logger.info(f"Audio downloaded successfully - Format: {audio_format}, Size: {len(audio_data)} bytes")
        
        # Create transcription request
        request = TranscriptionRequest(
            audio_base64=audio_data,
            audio_format=audio_format,
            model_name=model_name,
            language=language,
            return_timestamps=return_timestamps
        )
        
        # Process transcription
        return await transcribe_audio(request)
        
    except httpx.HTTPError as e:
        logger.error(f"Error downloading audio from URL: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to download audio: {str(e)}")
    except Exception as e:
        logger.error(f"URL transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"URL transcription failed: {str(e)}")


def infer_audio_format(url: str, content_type: str) -> AudioFormat:
    """Infer audio format from URL and content type."""
    # Check content type first
    if 'wav' in content_type or 'wave' in content_type:
        return AudioFormat.WAV
    elif 'mpeg' in content_type or 'mp3' in content_type:
        return AudioFormat.MP3
    elif 'flac' in content_type:
        return AudioFormat.FLAC
    elif 'mp4' in content_type or 'm4a' in content_type:
        return AudioFormat.M4A
    elif 'ogg' in content_type:
        return AudioFormat.OGG
    
    # Check URL extension
    url_lower = url.lower()
    if url_lower.endswith('.wav'):
        return AudioFormat.WAV
    elif url_lower.endswith('.mp3'):
        return AudioFormat.MP3
    elif url_lower.endswith('.flac'):
        return AudioFormat.FLAC
    elif url_lower.endswith('.m4a'):
        return AudioFormat.M4A
    elif url_lower.endswith('.ogg'):
        return AudioFormat.OGG
    
    # Default to WAV
    return AudioFormat.WAV


async def process_batch_transcription(
    job_id: str,
    audio_files: List[Dict],
    model_name: ModelName,
    language: Optional[str]
):
    """Process batch transcription in the background."""
    try:
        job = batch_jobs[job_id]
        
        # Process each file
        for i, audio_file in enumerate(audio_files):
            try:
                logger.info(f"Processing batch file {i+1}/{len(audio_files)} for job {job_id}")
                
                # Create transcription request
                request = TranscriptionRequest(
                    audio_base64=audio_file["audio_data"],
                    audio_format=audio_file["audio_format"],
                    model_name=model_name,
                    language=language,
                    return_timestamps=False  # Simplified for batch processing
                )
                
                # Run transcription in thread pool to avoid blocking
                result = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda: asyncio.run(transcribe_audio(request))
                )
                
                job["results"].append({
                    "index": i,
                    "filename": audio_file.get("filename", f"file_{i}"),
                    "text": result.text,
                    "confidence": result.confidence,
                    "processing_time": result.processing_time
                })
                
                job["completed_files"] += 1
                
            except Exception as e:
                logger.error(f"Error processing batch file {i}: {str(e)}")
                job["errors"].append({
                    "index": i,
                    "filename": audio_file.get("filename", f"file_{i}"),
                    "error": str(e)
                })
                job["completed_files"] += 1
        
        # Mark job as completed
        job["status"] = "completed"
        job["completed_at"] = datetime.now()
        
        logger.info(f"Batch transcription job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Batch transcription job {job_id} failed: {str(e)}")
        job["status"] = "failed"
        job["error"] = str(e)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8008,
        reload=True,
        log_level="info"
    )