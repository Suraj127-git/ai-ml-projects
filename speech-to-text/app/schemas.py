from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union
from datetime import datetime
from enum import Enum

class AudioFormat(str, Enum):
    """Supported audio formats"""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    M4A = "m4a"
    OGG = "ogg"

class ModelName(str, Enum):
    """Available speech recognition models"""
    WAV2VEC2_BASE = "wav2vec2-base"
    WAV2VEC2_LARGE = "wav2vec2-large"
    WHISPER_BASE = "whisper-base"
    WHISPER_SMALL = "whisper-small"
    WHISPER_MEDIUM = "whisper-medium"

class TranscriptionRequest(BaseModel):
    """Request for speech transcription"""
    audio_base64: str = Field(..., description="Base64 encoded audio file")
    audio_format: AudioFormat = Field(default=AudioFormat.WAV, description="Audio file format")
    model_name: ModelName = Field(default=ModelName.WAV2VEC2_BASE, description="Model to use for transcription")
    language: Optional[str] = Field(default=None, description="Language code (e.g., 'en', 'es', 'fr')")
    return_timestamps: bool = Field(default=False, description="Whether to return word-level timestamps")
    
    class Config:
        json_schema_extra = {
            "example": {
                "audio_base64": "UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIG2m98OScTgwOUarm7blmFgU7k9n1unEiBC13yO/eizEIHWq+8+OWT...",
                "audio_format": "wav",
                "model_name": "wav2vec2-base",
                "language": "en",
                "return_timestamps": False
            }
        }

class WordTimestamp(BaseModel):
    """Word-level timestamp information"""
    word: str = Field(..., description="The transcribed word")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    confidence: Optional[float] = Field(None, description="Confidence score for this word")

class TranscriptionSegment(BaseModel):
    """Transcription segment with optional timestamps"""
    text: str = Field(..., description="Transcribed text segment")
    start_time: Optional[float] = Field(None, description="Start time in seconds")
    end_time: Optional[float] = Field(None, description="End time in seconds")
    confidence: Optional[float] = Field(None, description="Confidence score for this segment")
    words: Optional[List[WordTimestamp]] = Field(None, description="Word-level timestamps")

class TranscriptionResponse(BaseModel):
    """Response for speech transcription"""
    text: str = Field(..., description="Complete transcribed text")
    segments: Optional[List[TranscriptionSegment]] = Field(None, description="Text segments with timestamps")
    model_name: str = Field(..., description="Model used for transcription")
    language: Optional[str] = Field(None, description="Detected or specified language")
    audio_duration: float = Field(..., description="Audio duration in seconds")
    processing_time: float = Field(..., description="Processing time in seconds")
    confidence: Optional[float] = Field(None, description="Overall confidence score")
    word_count: int = Field(..., description="Number of words transcribed")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of transcription")

class AudioInfo(BaseModel):
    """Audio file information"""
    format: str = Field(..., description="Audio format")
    duration: float = Field(..., description="Duration in seconds")
    sample_rate: int = Field(..., description="Sample rate in Hz")
    channels: int = Field(..., description="Number of audio channels")
    bit_depth: Optional[int] = Field(None, description="Bit depth")
    file_size: int = Field(..., description="File size in bytes")

class AudioAnalysisRequest(BaseModel):
    """Request for audio analysis"""
    audio_base64: str = Field(..., description="Base64 encoded audio file")
    audio_format: AudioFormat = Field(default=AudioFormat.WAV, description="Audio file format")

class AudioAnalysisResponse(BaseModel):
    """Response for audio analysis"""
    audio_info: AudioInfo = Field(..., description="Audio file information")
    is_speech_detected: bool = Field(..., description="Whether speech is detected in the audio")
    speech_quality_score: Optional[float] = Field(None, description="Speech quality score (0-1)")
    noise_level: Optional[float] = Field(None, description="Estimated noise level")
    recommended_models: List[str] = Field(..., description="Recommended models for this audio")

class BatchTranscriptionRequest(BaseModel):
    """Request for batch speech transcription"""
    audio_files: List[Dict[str, str]] = Field(..., description="List of audio files with base64 encoding and metadata")
    model_name: ModelName = Field(default=ModelName.WAV2VEC2_BASE, description="Model to use for transcription")
    language: Optional[str] = Field(default=None, description="Language code")
    return_timestamps: bool = Field(default=False, description="Whether to return word-level timestamps")

class BatchTranscriptionResponse(BaseModel):
    """Response for batch speech transcription"""
    transcriptions: List[TranscriptionResponse] = Field(..., description="Transcriptions for each audio file")
    total_files: int = Field(..., description="Total number of audio files processed")
    total_processing_time: float = Field(..., description="Total processing time in seconds")
    average_processing_time: float = Field(..., description="Average processing time per file")
    successful_transcriptions: int = Field(..., description="Number of successful transcriptions")
    failed_transcriptions: int = Field(..., description="Number of failed transcriptions")

class ModelTrainingRequest(BaseModel):
    """Request for model fine-tuning"""
    training_data_path: str = Field(..., description="Path to training data directory")
    base_model: ModelName = Field(default=ModelName.WAV2VEC2_BASE, description="Base model to fine-tune")
    language: str = Field(..., description="Target language for fine-tuning")
    epochs: int = Field(default=3, ge=1, le=10, description="Number of training epochs")
    learning_rate: float = Field(default=5e-5, gt=0.0, description="Learning rate for fine-tuning")
    batch_size: int = Field(default=8, ge=4, le=32, description="Batch size for training")

class ModelTrainingResponse(BaseModel):
    """Response for model fine-tuning"""
    message: str = Field(..., description="Training status message")
    model_name: str = Field(..., description="Name of the fine-tuned model")
    base_model: str = Field(..., description="Base model used for fine-tuning")
    language: str = Field(..., description="Target language")
    epochs_completed: int = Field(..., description="Number of epochs completed")
    training_samples: int = Field(..., description="Number of training samples used")
    training_time: float = Field(..., description="Training time in seconds")
    final_loss: float = Field(..., description="Final training loss")
    word_error_rate: Optional[float] = Field(None, description="Word error rate on validation set")

class SupportedLanguage(BaseModel):
    """Supported language information"""
    code: str = Field(..., description="Language code (e.g., 'en', 'es', 'fr')")
    name: str = Field(..., description="Language name")
    models_available: List[str] = Field(..., description="Available models for this language")
    confidence_threshold: float = Field(default=0.7, description="Recommended confidence threshold")

class ModelInfo(BaseModel):
    """Model information"""
    name: str = Field(..., description="Model name")
    description: str = Field(..., description="Model description")
    languages_supported: List[str] = Field(..., description="Supported languages")
    max_audio_duration: float = Field(..., description="Maximum audio duration in seconds")
    recommended_sample_rate: int = Field(..., description="Recommended sample rate in Hz")
    model_size_mb: float = Field(..., description="Model size in MB")
    inference_time_factor: float = Field(..., description="Relative inference time factor")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    models_loaded: List[str] = Field(..., description="List of loaded models")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in MB")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp")