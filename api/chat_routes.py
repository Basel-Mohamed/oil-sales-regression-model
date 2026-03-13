"""
chat_routes.py
Conversational AI agent for extracting Oil Sales data using Cohere.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import cohere
import json
import os
import traceback

from .ml_routes import pipeline

router = APIRouter()

# Load API key dynamically to prevent GitHub Push Protection blocks
co = cohere.Client(os.getenv("COHERE_API_KEY"))

sessions: Dict[str, Dict[str, Any]] = {}

# The complete list of features required by the Oil Sales Regression model
REQUIRED_FIELDS = [
    "city", "store_name", "manufacturer", "brand", "class", 
    "size", "price_bracket", "year", "month", "value_sales", "average_price"
]

class ChatRequest(BaseModel):
    message: str
    session_id: str

class ResetRequest(BaseModel):
    session_id: str

@router.post("/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id
    user_message = request.message

    if session_id not in sessions:
        sessions[session_id] = {
            "history": [],
            "extracted_data": {}
        }

    session = sessions[session_id]
    current_data = session["extracted_data"]
    
    missing_fields = [f for f in REQUIRED_FIELDS if f not in current_data]

    preamble = f"""You are a sales forecasting assistant gathering data to predict Oil Volume Sales.
    Extract product details from the user's message.
    
    Expected Fields: {REQUIRED_FIELDS}
    Currently Collected: {json.dumps(current_data)}
    Missing Fields: {missing_fields}

    CRITICAL INSTRUCTIONS FOR EXTRACTION:
    - "year" and "month" must be integers.
    - "value_sales" and "average_price" must be floats.
    - "class" refers to the product category/class.
    - All other fields are strings.

    Return ONLY a valid JSON object with:
    1. "extracted_fields": Dictionary of NEW data found.
    2. "agent_reply": Your conversational response asking for 1 or 2 missing fields.
    """

    try:
        response = co.chat(
            message=user_message,
            preamble=preamble,
            chat_history=session["history"],
            response_format={"type": "json_object"}
        )

        llm_output = json.loads(response.text)
        new_extracted = llm_output.get("extracted_fields", {})
        agent_reply = llm_output.get("agent_reply", "Could you provide more details?")

        session["history"].append({"role": "USER", "message": user_message})
        session["history"].append({"role": "CHATBOT", "message": agent_reply})
        
        # Update session with new extracted data
        session["extracted_data"].update(new_extracted)

        updated_missing = [f for f in REQUIRED_FIELDS if f not in session["extracted_data"]]

        # If all fields are collected, run the prediction
        if not updated_missing:
            clean_data = session["extracted_data"].copy()
            
            # Type-cast numeric fields to satisfy the ML model
            try:
                if "year" in clean_data: clean_data["year"] = int(clean_data["year"])
                if "month" in clean_data: clean_data["month"] = int(clean_data["month"])
                if "value_sales" in clean_data: clean_data["value_sales"] = float(clean_data["value_sales"])
                if "average_price" in clean_data: clean_data["average_price"] = float(clean_data["average_price"])
            except ValueError as e:
                print(f"Data casting error: {e}")

            try:
                # Trigger the Regression prediction
                prediction_result = pipeline.predict_single(clean_data)
                
                # Extract predicted volume (adjust key if your pipeline returns a dict)
                predicted_volume = 0
                if isinstance(prediction_result, dict):
                    predicted_volume = prediction_result.get("prediction", prediction_result.get("volume_sales", 0))
                else:
                    predicted_volume = float(prediction_result)
                
                # Format the response for the user
                final_text = f"📊 Sales Prediction Results for {clean_data.get('brand', 'Product')}\n\n"
                final_text += f"🏢 Manufacturer: {clean_data.get('manufacturer', 'Unknown')}\n"
                final_text += f"📍 City/Store: {clean_data.get('city', 'Unknown')} - {clean_data.get('store_name', 'Unknown')}\n"
                final_text += f"📅 Period: {clean_data.get('month', '')}/{clean_data.get('year', '')}\n\n"
                
                # Using comma formatting for large volume numbers
                final_text += f"✨ **Predicted Volume Sales: {predicted_volume:,.2f} units**\n\n"
                
                final_text += "Would you like to forecast another product?"

                return {
                    "status": "success",
                    "response": final_text,
                    "session_id": session_id,
                    "prediction_data": prediction_result,
                    "extracted_data": session["extracted_data"] # Added here to ensure UI stays synced at the end
                }
                
            except Exception as ve:
                return {
                    "status": "error",
                    "response": f"⚠️ I collected all the data, but the model rejected it: {str(ve)}. Please reset the chat and try again.",
                    "session_id": session_id
                }

        # For ongoing conversations, return the extracted data progressively!
        return {
            "status": "success",
            "response": agent_reply,
            "session_id": session_id,
            "extracted_data": session["extracted_data"] # <--- THE CRITICAL FIX IS HERE
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/chat/reset")
async def reset_chat(request: ResetRequest):
    session_id = request.session_id
    if session_id in sessions:
        sessions[session_id] = {
            "history": [],
            "extracted_data": {}
        }
    return {"status": "success", "message": "Session reset"}