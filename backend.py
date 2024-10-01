from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# CORS Middleware to allow the frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define the request body model
class Message(BaseModel):
    message: str

# Define a simple chatbot endpoint
@app.post("/chatbot")
async def chatbot(message: Message):
    # For demonstration, we'll just return a simple response
    response_text = f"You said: {message.message}"
    return {"response": response_text}

# You can now run the FastAPI server using `uvicorn`:
# uvicorn backend:app --host 127.0.0.1 --port 5000 --reload


# Define a simple request body
# class Message(BaseModel):
#     text: str

# @app.post("/chatbot")
# async def chatbot(message: Message):
#     return {"response": f"You said: {message.text}"}
