# src/serve_validated.py

import mlflow
import mlflow.sklearn
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.data_validation import validate_transaction
from src.feast_feature import get_online_features

mlflow.set_tracking_uri("http://localhost:5000")

print("Loading model from MLflow Model Registry...")

try:
    model = mlflow.sklearn.load_model("models:/fraud-detection-model@champion")
    print("Successfully loaded champion model from MLflow!")
except Exception as e:
    print(f"Error loading from MLflow: {e}")
    print("Make sure you've assigned the @champion alias to a model in the MLflow UI")
    raise

with open("models/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
print("Encoder loaded successfully!")

app = FastAPI(
    title="Fraud Detection API",
    version="3.0.0"
)

class Transaction(BaseModel):
    amount: float = Field(..., description="Transaction amount (must be positive)", example=150.00)
    hour: int = Field(..., description="Hour of day (0-23)", example=14)
    day_of_week: int = Field(..., description="Day of week (0=Mon, 6=Sun)", example=3)
    merchant_category: str = Field(..., description="Merchant type", example="online")

class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    model_source: str = "MLflow Production"
    validation_passed: bool = True

class ValidationErrorResponse(BaseModel):
    detail: dict

@app.post("/predict", response_model=PredictionResponse, responses={400: {"model": ValidationErrorResponse}})
def predict(tx: Transaction):
    """
    Predict whether a transaction is fraudulent.
    
    Input is validated before prediction. Invalid inputs return HTTP 400.
    """
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
    
    feast_features = get_online_features(data["merchant_category"])

    data["merchant_encoded"] = encoder.transform([data["merchant_category"]])[0]
    X = [[
        data["amount"],
        data["hour"],
        data["day_of_week"],
        data["merchant_encoded"],
        feast_features["merchant_avg_amount"],      # from Feast
        feast_features["merchant_tx_count"],         # from Feast
        feast_features["merchant_fraud_rate"],       # from Feast
    ]]
    
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    
    return PredictionResponse(
        is_fraud=bool(pred),
        fraud_probability=round(float(prob), 4),
        validation_passed=True,
        model_source="MLflow Production"
    )

@app.get("/health")
def health():
    return {"status": "healthy", "validation": "enabled"}

@app.get("/model-info")
def model_info():
    return {
        "registry": "MLflow",
        "model_name": "fraud-detection-model",
        "alias": "production",
        "tracking_uri": "http://localhost:5000"
    }