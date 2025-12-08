import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from typing import Optional
import logging

from .schemas import ModelName

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextToSQLModel:
    """Text-to-SQL model using T5 architecture"""
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.current_model_name = None
        self.is_loaded = False
        
    def load_model(self, model_name: ModelName = ModelName.T5_SMALL):
        """Load the T5 model for text-to-SQL generation"""
        try:
            logger.info(f"Loading {model_name.value} model...")
            
            self.tokenizer = T5Tokenizer.from_pretrained(model_name.value)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name.value)
            self.current_model_name = model_name
            self.is_loaded = True
            
            logger.info(f"Model {model_name.value} loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model {model_name.value}: {e}")
            raise
    
    def generate_sql(
        self, 
        text: str, 
        model_name: Optional[ModelName] = None,
        max_length: int = 512,
        temperature: float = 0.7
    ) -> str:
        """Generate SQL query from natural language text"""
        
        if not self.is_loaded or (model_name and model_name != self.current_model_name):
            self.load_model(model_name or ModelName.T5_SMALL)
        
        try:
            # Prepare input text with SQL generation prompt
            input_text = f"translate English to SQL: {text}"
            
            # Tokenize input
            inputs = self.tokenizer.encode(
                input_text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            )
            
            # Generate SQL
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=temperature,
                    num_beams=4,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode generated SQL
            sql_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the generated SQL
            sql_query = sql_query.strip()
            if sql_query.lower().startswith("translate english to sql:"):
                sql_query = sql_query[len("translate english to sql:"):].strip()
            
            return sql_query
            
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            raise
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None and self.tokenizer is not None