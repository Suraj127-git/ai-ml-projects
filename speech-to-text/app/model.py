import torch
import torchaudio
import librosa
import soundfile as sf
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, WhisperProcessor, WhisperForConditionalGeneration
import base64
import io
from pydub import AudioSegment
import tempfile
import os
import time
from datetime import datetime
import logging

from .schemas import AudioFormat, ModelName, TranscriptionSegment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeechToTextModel:
    """
    Speech-to-Text model supporting both Wav2Vec2 and Whisper architectures.
    Handles audio preprocessing, model loading, and transcription with optional timestamps.
    """
    
    def __init__(self):
        self.models = {}
        self.processors = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    def load_model(self, model_name: ModelName) -> bool:
        """
        Load the specified speech recognition model and processor.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if model_name in self.models:
                logger.info(f"Model {model_name} already loaded")
                return True
            
            logger.info(f"Loading model: {model_name}")
            
            if model_name in [ModelName.WAV2VEC2_BASE, ModelName.WAV2VEC2_LARGE]:
                # Load Wav2Vec2 model
                model_id = {
                    ModelName.WAV2VEC2_BASE: "facebook/wav2vec2-base-960h",
                    ModelName.WAV2VEC2_LARGE: "facebook/wav2vec2-large-960h"
                }[model_name]
                
                processor = Wav2Vec2Processor.from_pretrained(model_id)
                model = Wav2Vec2ForCTC.from_pretrained(model_id)
                
            elif model_name in [ModelName.WHISPER_BASE, ModelName.WHISPER_SMALL, ModelName.WHISPER_MEDIUM]:
                # Load Whisper model
                model_id = {
                    ModelName.WHISPER_BASE: "openai/whisper-base",
                    ModelName.WHISPER_SMALL: "openai/whisper-small",
                    ModelName.WHISPER_MEDIUM: "openai/whisper-medium"
                }[model_name]
                
                processor = WhisperProcessor.from_pretrained(model_id)
                model = WhisperForConditionalGeneration.from_pretrained(model_id)
                
            else:
                logger.error(f"Unsupported model: {model_name}")
                return False
            
            # Move model to appropriate device
            model = model.to(self.device)
            model.eval()
            
            self.models[model_name] = model
            self.processors[model_name] = processor
            
            logger.info(f"Successfully loaded model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            return False
    
    def preprocess_audio(self, audio_data: bytes, audio_format: AudioFormat, target_sample_rate: int = 16000) -> np.ndarray:
        """
        Preprocess audio data for speech recognition.
        
        Args:
            audio_data: Raw audio bytes
            audio_format: Format of the audio data
            target_sample_rate: Target sample rate for the model
            
        Returns:
            Preprocessed audio array
        """
        try:
            # Convert base64 to audio if needed
            if isinstance(audio_data, str):
                audio_data = base64.b64decode(audio_data)
            
            # Create temporary file for audio processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{audio_format.value}') as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            try:
                # Load audio with librosa
                audio_array, sample_rate = librosa.load(temp_file_path, sr=target_sample_rate)
                
                # Normalize audio
                audio_array = audio_array / np.max(np.abs(audio_array))
                
                # Remove silence from beginning and end
                audio_array, _ = librosa.effects.trim(audio_array, top_db=20)
                
                return audio_array
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            logger.error(f"Error preprocessing audio: {str(e)}")
            # Fallback: try with pydub
            try:
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format=audio_format.value)
                audio_segment = audio_segment.set_frame_rate(target_sample_rate).set_channels(1)
                
                # Convert to numpy array
                audio_array = np.array(audio_segment.get_array_of_samples())
                audio_array = audio_array.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
                
                return audio_array
                
            except Exception as e2:
                logger.error(f"Fallback audio processing failed: {str(e2)}")
                raise ValueError(f"Failed to process audio data: {str(e)}")
    
    def transcribe_with_wav2vec2(self, audio_array: np.ndarray, model_name: ModelName, language: Optional[str] = None) -> Dict:
        """
        Transcribe audio using Wav2Vec2 model.
        
        Args:
            audio_array: Preprocessed audio array
            model_name: Name of the Wav2Vec2 model
            language: Language code (ignored for Wav2Vec2)
            
        Returns:
            Transcription result dictionary
        """
        try:
            processor = self.processors[model_name]
            model = self.models[model_name]
            
            # Process audio through the model
            inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                logits = model(**inputs).logits
            
            # Decode the predictions
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.decode(predicted_ids[0])
            
            # Calculate confidence (average probability of predicted tokens)
            probs = torch.softmax(logits, dim=-1)
            predicted_probs = torch.gather(probs, -1, predicted_ids.unsqueeze(-1)).squeeze(-1)
            confidence = float(torch.mean(predicted_probs[predicted_probs > 0]).item())
            
            return {
                "text": transcription.strip(),
                "confidence": confidence,
                "language": "en"  # Wav2Vec2 models are primarily English
            }
            
        except Exception as e:
            logger.error(f"Error in Wav2Vec2 transcription: {str(e)}")
            raise ValueError(f"Transcription failed: {str(e)}")
    
    def transcribe_with_whisper(self, audio_array: np.ndarray, model_name: ModelName, language: Optional[str] = None, return_timestamps: bool = False) -> Dict:
        """
        Transcribe audio using Whisper model.
        
        Args:
            audio_array: Preprocessed audio array
            model_name: Name of the Whisper model
            language: Language code (e.g., 'en', 'es', 'fr')
            return_timestamps: Whether to return word-level timestamps
            
        Returns:
            Transcription result dictionary
        """
        try:
            processor = self.processors[model_name]
            model = self.models[model_name]
            
            # Process audio through the model
            inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate transcription
            forced_decoder_ids = None
            if language:
                forced_decoder_ids = processor.get_decoder_prompt_ids(language=language)
            
            with torch.no_grad():
                if return_timestamps:
                    predicted_ids = model.generate(**inputs, return_timestamps=True, forced_decoder_ids=forced_decoder_ids)
                else:
                    predicted_ids = model.generate(**inputs, forced_decoder_ids=forced_decoder_ids)
            
            # Decode the predictions
            transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
            
            result = {
                "text": transcription.strip(),
                "language": language or "auto"
            }
            
            # Extract segments if timestamps are requested
            if return_timestamps and hasattr(predicted_ids, 'timestamp_tokens'):
                segments = self._extract_whisper_segments(predicted_ids, processor)
                result["segments"] = segments
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Whisper transcription: {str(e)}")
            raise ValueError(f"Transcription failed: {str(e)}")
    
    def _extract_whisper_segments(self, predicted_ids, processor) -> List[Dict]:
        """
        Extract timestamped segments from Whisper output.
        
        Args:
            predicted_ids: Model output tensor
            processor: Whisper processor
            
        Returns:
            List of segment dictionaries with timestamps
        """
        try:
            # This is a simplified implementation
            # In practice, you might need more sophisticated timestamp extraction
            segments = []
            
            # Decode with timestamps
            decoded_with_timestamps = processor.decode(predicted_ids[0], skip_special_tokens=False)
            
            # Parse timestamps (this is a basic implementation)
            import re
            timestamp_pattern = r'<\|(\d+\.\d+)\|>'
            timestamps = re.findall(timestamp_pattern, decoded_with_timestamps)
            
            # Clean text and create segments
            clean_text = re.sub(timestamp_pattern, '', decoded_with_timestamps).strip()
            words = clean_text.split()
            
            # Simple word-level timestamp assignment
            if timestamps and len(timestamps) >= 2:
                start_time = float(timestamps[0])
                end_time = float(timestamps[-1])
                
                # Distribute words across time segments
                time_per_word = (end_time - start_time) / max(len(words), 1)
                
                for i, word in enumerate(words):
                    word_start = start_time + (i * time_per_word)
                    word_end = word_start + time_per_word
                    
                    segments.append({
                        "text": word,
                        "start": word_start,
                        "end": word_end
                    })
            
            return segments
            
        except Exception as e:
            logger.error(f"Error extracting segments: {str(e)}")
            return []
    
    def transcribe_audio(self, audio_data: Union[str, bytes], audio_format: AudioFormat, 
                        model_name: ModelName, language: Optional[str] = None, 
                        return_timestamps: bool = False) -> Dict:
        """
        Main transcription method that handles audio preprocessing and model selection.
        
        Args:
            audio_data: Base64 encoded audio string or raw audio bytes
            audio_format: Format of the audio data
            model_name: Name of the model to use
            language: Language code (optional)
            return_timestamps: Whether to return word-level timestamps
            
        Returns:
            Complete transcription result
        """
        start_time = time.time()
        
        try:
            # Load model if not already loaded
            if not self.load_model(model_name):
                raise ValueError(f"Failed to load model: {model_name}")
            
            # Preprocess audio
            logger.info("Preprocessing audio...")
            audio_array = self.preprocess_audio(audio_data, audio_format)
            audio_duration = len(audio_array) / 16000  # Assuming 16kHz sample rate
            
            # Perform transcription based on model type
            logger.info(f"Transcribing with {model_name}...")
            
            if model_name in [ModelName.WAV2VEC2_BASE, ModelName.WAV2VEC2_LARGE]:
                result = self.transcribe_with_wav2vec2(audio_array, model_name, language)
            else:  # Whisper models
                result = self.transcribe_with_whisper(audio_array, model_name, language, return_timestamps)
            
            processing_time = time.time() - start_time
            
            # Create comprehensive result
            final_result = {
                "text": result["text"],
                "model_name": model_name.value,
                "language": result.get("language", language or "unknown"),
                "audio_duration": audio_duration,
                "processing_time": processing_time,
                "confidence": result.get("confidence"),
                "word_count": len(result["text"].split()),
                "timestamp": datetime.now()
            }
            
            # Add segments if available
            if "segments" in result:
                final_result["segments"] = result["segments"]
            
            logger.info(f"Transcription completed in {processing_time:.2f} seconds")
            return final_result
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise ValueError(f"Transcription failed: {str(e)}")
    
    def batch_transcribe(self, audio_files: List[Dict], model_name: ModelName, 
                          language: Optional[str] = None) -> List[Dict]:
        """
        Batch transcribe multiple audio files.
        
        Args:
            audio_files: List of dictionaries with 'audio_data' and 'audio_format'
            model_name: Model to use for transcription
            language: Language code (optional)
            
        Returns:
            List of transcription results
        """
        results = []
        
        for i, audio_file in enumerate(audio_files):
            try:
                logger.info(f"Processing batch item {i+1}/{len(audio_files)}")
                
                result = self.transcribe_audio(
                    audio_data=audio_file["audio_data"],
                    audio_format=audio_file["audio_format"],
                    model_name=model_name,
                    language=language
                )
                
                results.append({
                    "index": i,
                    "success": True,
                    "result": result
                })
                
            except Exception as e:
                logger.error(f"Batch processing failed for item {i}: {str(e)}")
                results.append({
                    "index": i,
                    "success": False,
                    "error": str(e)
                })
        
        return results
    
    def get_supported_languages(self, model_name: ModelName) -> List[str]:
        """
        Get list of supported languages for a given model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of language codes
        """
        if model_name in [ModelName.WAV2VEC2_BASE, ModelName.WAV2VEC2_LARGE]:
            return ["en"]  # Wav2Vec2 models are primarily English
        else:  # Whisper models
            return ["en", "es", "fr", "de", "it", "pt", "nl", "pl", "tr", "ru", 
                   "zh", "ja", "ko", "ar", "hi", "vi", "th", "cs", "da", "fi", 
                   "el", "he", "hu", "id", "ms", "no", "ro", "sk", "sv", "uk"]
    
    def get_model_info(self, model_name: ModelName) -> Dict:
        """
        Get information about a loaded model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information dictionary
        """
        if model_name not in self.models:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "model_name": model_name.value,
            "architecture": "Wav2Vec2" if "wav2vec2" in model_name.value else "Whisper",
            "supported_languages": self.get_supported_languages(model_name),
            "device": str(self.device)
        }