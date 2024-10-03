from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

class Message(BaseModel):
    message: str

@app.post("/chatbot")
async def chatbot(message: Message):
    response_text = f"You said: {message.message}"
    return {"response": response_text}

