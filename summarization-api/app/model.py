from transformers import pipeline

_sum = None

def load_model():
    global _sum
    _sum = pipeline("summarization")

def summarize(text: str):
    out = _sum(text, max_length=130, min_length=30, do_sample=False)
    return out[0]['summary_text']
