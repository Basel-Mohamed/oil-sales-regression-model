"""
app.py
Main entry point for the Oil Sales Prediction API.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Import the modular routers
from api.ml_routes import router as ml_router
from api.chat_routes import router as chat_router

app = FastAPI(
    title="Oil Sales Prediction & Chatbot API",
    description="ML API and Conversational Agent for predicting oil volume sales",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register the routers
app.include_router(ml_router)
app.include_router(chat_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)