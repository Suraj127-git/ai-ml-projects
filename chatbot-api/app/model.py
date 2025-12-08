from transformers import pipeline

_gen = None

def load_model():
    global _gen
    _gen = pipeline("text-generation", model="distilgpt2")

def reply(message: str):
    out = _gen(message, max_length=60, num_return_sequences=1)
    return out[0]['generated_text']
