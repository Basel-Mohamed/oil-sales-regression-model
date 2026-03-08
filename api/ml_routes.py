"""
ml_routes.py
Direct machine learning endpoints for the Oil Sales Regression Model.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import os
import sys

# Ensure we can import from src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.inference import PredictionPipeline

router = APIRouter()

# Initialize the pipeline once at startup
try:
    pipeline = PredictionPipeline(
        model_path=os.path.join(BASE_DIR, 'models', 'random_forest_model.pkl'),
        scaler_path=os.path.join(BASE_DIR, 'models', 'scaler.pkl'),
        encoders_path=os.path.join(BASE_DIR, 'models', 'encoders.pkl')
    )
    print("Regression model loaded successfully")
except Exception as e:
    print(f"Warning: Could not load regression model - {str(e)}")

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
        populate_by_name = True

@router.get("/")
def read_root():
    return {"message": "Oil Sales Prediction API", "status": "active"}

@router.get("/health")
def health_check():
    return {"status": "healthy", "model": "loaded"}

@router.post("/predict")
def predict_single(request: PredictionRequest):
    """Predict volume sales for a single record via direct JSON payload"""
    try:
        # Extract dictionary, map 'class_' back to 'class' for the model
        record = request.dict(by_alias=True)
        if 'class_' in record:
            record['class'] = record.pop('class_')
            
        result = pipeline.predict_single(record)
        return {"status": "success", "prediction": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))