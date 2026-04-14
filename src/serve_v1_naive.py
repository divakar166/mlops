import pickle
from fastapi import FastAPI
from pydantic import BaseModel, Field

print("Loading model...")
with open("models/model.pkl", "rb") as f:
    model, encoder = pickle.load(f)
print("Model loaded successfully!")

# FastAPI Application
app = FastAPI(
    title="Fraud Detection API",
    version="1.0.0"
)

class Transaction(BaseModel):
    amount: float = Field(
        ..., 
        description="Transaction amount in dollars",
        example=150.00
    )
    hour: int = Field(
        ..., 
        description="Hour of the day (0-23)",
        example=14
    )
    day_of_week: int = Field(
        ..., 
        description="Day of week (0=Monday, 6=Sunday)",
        example=3
    )
    merchant_category: str = Field(
        ..., 
        description="Type of merchant",
        example="online"
    )

class PredictionResponse(BaseModel):
    is_fraud: bool = Field(description="Whether the transaction is predicted as fraud")
    fraud_probability: float = Field(description="Probability of fraud (0.0 to 1.0)")
    
@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    """
    Predict whether a transaction is fraudulent.
    
    Takes transaction details and returns a fraud prediction
    along with the probability score.
    """
    # Convert the request to a dictionary
    data = transaction.model_dump()
    
    # Encode the merchant category using the same encoder from training
    # This ensures consistency between training and serving
    try:
        data["merchant_encoded"] = encoder.transform([data["merchant_category"]])[0]
    except ValueError:
        # Handle unknown merchant categories
        data["merchant_encoded"] = 0
    
    # Prepare features in the same order as training
    X = [[
        data["amount"],
        data["hour"],
        data["day_of_week"],
        data["merchant_encoded"]
    ]]
    
    # Get prediction and probability
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]
    
    return PredictionResponse(
        is_fraud=bool(prediction),
        fraud_probability=round(float(probability), 4)
    )

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.get("/")
def root():
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }