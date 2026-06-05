from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {
        "message": "RAG API Running"
    }

@app.post("/chat")
def chat(request: ChatRequest):

    result = generator(
        request.message,
        max_new_tokens=100
    )

    answer = result[0]["generated_text"]

    return {
        "response": answer
    }