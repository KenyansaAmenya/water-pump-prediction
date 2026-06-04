# importing libraries

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the model and label encoder
model_pipeline = joblib.load("../models/best_xgb_model.joblib")

label_encoder = joblib.load("../models/label_encoder.joblib")

# creating the FastAPI app
app = FastAPI(
    title="Water Pump Status Prediction API",
    description="API to predict the status of water pumps based on input features.",
    version="1.0"
    )

# input schema
class PumpData(BaseModel):
        amount_tsh: float
        gps_height: int
        population: int
        age: float
        month_recorded: int
        permit: bool
        waterpoint_type_group: str
        source_class: str
        quantity: str
        quality_group: str
        payment_type: str
        management_group: str
        extraction_type_class: str
        region: str
        basin: str

# Health check endpoint
@app.get("/")
def home():
    return {"message": "Welcome to the Water Pump Status Prediction API!"}

# Prediction endpoint
@app.post("/predict")
def predict(data: PumpData):
    input_df = pd.DataFrame([data.model_dump()])   
    prediction = model_pipeline.predict(input_df)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    return {"predicted_status": predicted_label} 

