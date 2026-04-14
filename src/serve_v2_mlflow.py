# src/serve_mlflow.py

import mlflow
import mlflow.sklearn
import pickle
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.data_validation import validate_transaction

mlflow.set_tracking_uri("http://localhost:5000")

print("Loading model from MLflow Model Registry...")

try:
    model = mlflow.sklearn.load_model("models:/fraud-detection-model@champion")
    print("Successfully loaded champion model from MLflow!")
except Exception as e:
    print(f"Error loading from MLflow: {e}")
    print("Make sure you've assigned the @champion alias to a model in the MLflow UI")
    raise

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
print("Encoder loaded successfully!")

app = FastAPI(
    title="Fraud Detection API (MLflow)",
    version="2.0.0"
)

class Transaction(BaseModel):
    amount: float = Field(..., description="Transaction amount in dollars", example=150.00)
    hour: int = Field(..., description="Hour of the day (0-23)", example=14)
    day_of_week: int = Field(..., description="Day of week (0=Monday, 6=Sunday)", example=3)
    merchant_category: str = Field(..., description="Type of merchant", example="online")

class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    model_source: str = "MLflow Production"

@app.post("/predict", response_model=PredictionResponse)
def predict(tx: Transaction):
    """Predict whether a transaction is fraudulent using the champion model."""
    data = tx.model_dump()

    validation = validate_transaction(data)
    
    if not validation["valid"]:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Validation failed",
                "errors": validation["errors"],
                "input": data
            }
        )
    
    data["merchant_encoded"] = encoder.transform([data["merchant_category"]])[0]
    X = [[data["amount"], data["hour"], data["day_of_week"], data["merchant_encoded"]]]
    
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    
    return PredictionResponse(
        is_fraud=bool(pred),
        fraud_probability=round(float(prob), 4),
        model_source="MLflow Production"
    )

@app.get("/health")
def health():
    return {"status": "healthy", "model_source": "MLflow Registry"}

@app.get("/model-info")
def model_info():
    """Get information about the currently loaded model."""
    return {
        "registry": "MLflow",
        "model_name": "fraud-detection-model",
        "alias": "champion",
        "tracking_uri": "http://localhost:5000"
    }