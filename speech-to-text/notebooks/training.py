"""
Speech-to-Text Model Training Script
This script demonstrates how to train speech-to-text models using Wav2Vec2 and Whisper
"""

import torch
import torchaudio
import librosa
import numpy as np
import pandas as pd
from transformers import (
    Wav2Vec2ForCTC, 
    Wav2Vec2Processor, 
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    TrainingArguments,
    Trainer
)
from datasets import Dataset, Audio
import soundfile as sf
import os
import logging
from typing import Dict, List, Optional, Tuple
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeechToTextTrainer:
    """Trainer class for speech-to-text models"""
    
    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h"):
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self):
        """Load the speech-to-text model and processor"""
        logger.info(f"Loading model: {self.model_name}")
        
        if "whisper" in self.model_name.lower():
            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
        else:
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
            
        self.model.to(self.device)
        logger.info("Model loaded successfully")
        
    def preprocess_audio(self, audio_path: str, target_sample_rate: int = 16000) -> np.ndarray:
        """Preprocess audio file to the required format"""
        try:
            # Load audio
            speech, sample_rate = librosa.load(audio_path, sr=target_sample_rate)
            
            # Normalize audio
            speech = (speech - speech.mean()) / speech.std()
            
            return speech
            
        except Exception as e:
            logger.error(f"Error preprocessing audio {audio_path}: {e}")
            raise
            
    def prepare_dataset(self, data: List[Dict]) -> Dataset:
        """Prepare dataset for training"""
        logger.info("Preparing dataset...")
        
        dataset_dict = {
            "audio": [],
            "text": []
        }
        
        for item in data:
            audio_path = item.get("audio_path")
            text = item.get("text", "")
            
            if audio_path and os.path.exists(audio_path):
                try:
                    audio_data = self.preprocess_audio(audio_path)
                    dataset_dict["audio"].append(audio_data)
                    dataset_dict["text"].append(text)
                except Exception as e:
                    logger.warning(f"Skipping item due to error: {e}")
                    
        return Dataset.from_dict(dataset_dict)
        
    def train(self, train_data: List[Dict], val_data: List[Dict], output_dir: str = "./models"):
        """Train the speech-to-text model"""
        logger.info("Starting training...")
        
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_data)
        val_dataset = self.prepare_dataset(val_data)
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            fp16=torch.cuda.is_available(),
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.processor,
        )
        
        # Train model
        trainer.train()
        
        # Save model
        trainer.save_model(output_dir)
        self.processor.save_pretrained(output_dir)
        
        logger.info(f"Training completed. Model saved to {output_dir}")
        
    def evaluate(self, test_data: List[Dict]) -> Dict:
        """Evaluate the trained model"""
        logger.info("Evaluating model...")
        
        test_dataset = self.prepare_dataset(test_data)
        
        # Create evaluation trainer
        eval_trainer = Trainer(
            model=self.model,
            args=TrainingArguments(output_dir="./eval"),
            eval_dataset=test_dataset,
            tokenizer=self.processor,
        )
        
        # Evaluate
        metrics = eval_trainer.evaluate()
        
        logger.info(f"Evaluation completed. Metrics: {metrics}")
        return metrics

def main():
    """Main training function"""
    logger.info("Starting speech-to-text training...")
    
    # Example training data (replace with your actual data)
    train_data = [
        {"audio_path": "audio1.wav", "text": "Hello world"},
        {"audio_path": "audio2.wav", "text": "How are you today"},
        # Add more training data...
    ]
    
    val_data = [
        {"audio_path": "val1.wav", "text": "Good morning"},
        {"audio_path": "val2.wav", "text": "Thank you very much"},
        # Add more validation data...
    ]
    
    # Initialize trainer
    trainer = SpeechToTextTrainer("facebook/wav2vec2-base-960h")
    
    # Load model
    trainer.load_model()
    
    # Train model
    trainer.train(train_data, val_data, output_dir="./speech_to_text_model")
    
    # Evaluate model
    metrics = trainer.evaluate(val_data)
    
    logger.info("Training completed successfully!")
    logger.info(f"Final metrics: {json.dumps(metrics, indent=2)}")

if __name__ == "__main__":
    main()