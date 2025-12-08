from fastapi import FastAPI
from app.schemas import ChatRequest, ChatResponse
from app.model import load_model, reply

app = FastAPI(title="Chatbot API")

@app.on_event("startup")
async def startup():
    load_model()

@app.get("/")
def root():
    return {"message": "Chatbot API"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    r = reply(req.message)
    return ChatResponse(reply=r)
