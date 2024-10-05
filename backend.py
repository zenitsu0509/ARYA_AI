from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # Adjust origins in production
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Pydantic model for request validation
class Message(BaseModel):
    message: str

# POST route for chatbot
@app.post("/chatbot")
async def chatbot(message: Message):
    try:
        response_text = f"You said: {message.message}"
        return {"response": response_text}
    except Exception as e:
        return {"error": str(e)}, 500
