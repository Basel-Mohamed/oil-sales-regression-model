from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import io
import sys
sys.path.append('..')
from src.inference import PredictionPipeline

app = FastAPI(title="Oil Sales Prediction API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
pipeline = PredictionPipeline(
    model_path='../models/random_forest_model.pkl',
    scaler_path='../models/scaler.pkl',
    encoders_path='../models/encoders.pkl'
)

class PredictionRequest(BaseModel):
    city: str
    store_name: str
    manufacturer: str
    brand: str
    class_: str = Field(alias="class")
    size: str
    price_bracket: str
    year: int
    month: int
    value_sales: float
    average_price: float
    
    class Config:
        fields = {'class_': 'class'}

@app.get("/")
def read_root():
    return {"message": "Oil Sales Prediction API", "status": "active"}

@app.post("/predict")
def predict_single(request: PredictionRequest):
    """Predict volume sales for a single record"""
    try:
        record = request.dict()
        record['class'] = record.pop('class_')
        result = pipeline.predict_single(record)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "loaded"}