import time
import uuid
import logging
import pickle
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.data_validation import validate_transaction
from src.feast_feature import get_online_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("fraud_detection")

class AppState:
    model = None
    encoder = None
    valid_categories: set[str] = set()
    model_version: str = "unknown"
    loaded_at: float = 0.0

state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    logger.info("Starting up fraud-detection service …")
    
    tracking_uri = "http://localhost:5000"
    model_uri    = "models:/fraud-detection-model@champion"
    mlflow.set_tracking_uri(tracking_uri)
    logger.info("MLflow tracking URI set to %s", tracking_uri)

    try:
        state.model = mlflow.sklearn.load_model(model_uri)
        logger.info("Champion model loaded from MLflow registry (%s)", model_uri)
    except Exception:
        logger.exception("Failed to load model from MLflow — aborting startup")
        raise

    try:
        client = mlflow.MlflowClient()
        mv = client.get_model_version_by_alias("fraud-detection-model", "champion")
        state.model_version = mv.version
        logger.info("Model version: %s (run_id=%s)", mv.version, mv.run_id)
    except Exception:
        logger.warning("Could not fetch model version metadata — continuing anyway")

    encoder_path = "models/encoder.pkl"
    try:
        with open(encoder_path, "rb") as f:
            state.encoder = pickle.load(f)
        state.valid_categories = set(state.encoder.classes_)
        logger.info(
            "Encoder loaded from %s (%d categories)",
            encoder_path,
            len(state.valid_categories),
        )
    except FileNotFoundError:
        logger.exception("Encoder file not found at %s", encoder_path)
        raise
    except Exception:
        logger.exception("Failed to load encoder")
        raise

    state.loaded_at = time.time()
    elapsed = time.perf_counter() - t0
    logger.info("Startup complete in %.3fs", elapsed)

    yield

    logger.info("Shutting down fraud-detection service …")
    state.model   = None
    state.encoder = None
    logger.info("Resources released. Goodbye.")

app = FastAPI(
    title="Fraud Detection API",
    version="3.0.0",
    description="Real-time transaction fraud scoring powered by MLflow.",
    lifespan=lifespan,
)

# Request-ID + latency middleware
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    t0 = time.perf_counter()

    logger.info(
        "REQUEST  id=%s method=%s path=%s",
        request_id, request.method, request.url.path,
    )

    response = await call_next(request)

    elapsed_ms = (time.perf_counter() - t0) * 1_000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "RESPONSE id=%s status=%d latency_ms=%.2f",
        request_id, response.status_code, elapsed_ms,
    )
    return response

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception("Unhandled error [request_id=%s]", request_id)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "request_id": request_id,
        },
    )

# Schemas

class Transaction(BaseModel):
    amount: float = Field(..., description="Transaction amount (must be positive)", json_schema_extra={"example": 150.00})
    hour: int = Field(..., description="Hour of day (0-23)", json_schema_extra={"example": 14})
    day_of_week: int = Field(..., description="Day of week (0=Mon, 6=Sun)", json_schema_extra={"example": 3})
    merchant_category: str = Field(..., description="Merchant type", json_schema_extra={"example": "online"})


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    model_source: str = "MLflow Production"
    model_version: str
    validation_passed: bool = True
    request_id: str


class ValidationErrorResponse(BaseModel):
    detail: dict

# Routes

@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ValidationErrorResponse}},
    tags=["Prediction"],
)
def predict(tx: Transaction, request: Request):
    """
    Score a transaction for fraud.

    - Validates input fields before inference.
    - Enriches features from the Feast online store.
    - Returns fraud probability from the MLflow champion model.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    data = tx.model_dump()
    logger.info("Prediction request [request_id=%s] payload=%s", request_id, data)

    # Validation
    validation = validate_transaction(data, state.valid_categories)
    if not validation["valid"]:
        logger.warning(
            "Validation failed [request_id=%s] errors=%s",
            request_id, validation["errors"],
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Validation failed",
                "errors": validation["errors"],
                "input": data,
            },
        )

    # Feature engineering
    try:
        feast_features = get_online_features(data["merchant_category"])
    except Exception:
        logger.exception("Feast feature fetch failed [request_id=%s]", request_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feature store unavailable — please retry.",
        )

    try:
        data["merchant_encoded"] = state.encoder.transform([data["merchant_category"]])[0]
    except Exception:
        logger.exception("Encoder transform failed [request_id=%s]", request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Feature encoding error.",
        )

    X = pd.DataFrame([{
        "amount":            data["amount"],
        "hour":              data["hour"],
        "day_of_week":       data["day_of_week"],
        "merchant_encoded":  data["merchant_encoded"],
        "avg_amount":        feast_features["merchant_avg_amount"],
        "transaction_count": feast_features["merchant_tx_count"],
        "fraud_rate":        feast_features["merchant_fraud_rate"],
    }])

    # Inference
    try:
        pred = state.model.predict(X)[0]
        prob = float(state.model.predict_proba(X)[0][1])
    except Exception:
        logger.exception("Model inference failed [request_id=%s]", request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model inference error.",
        )

    logger.info(
        "Prediction result [request_id=%s] is_fraud=%s probability=%.4f",
        request_id, bool(pred), prob,
    )

    return PredictionResponse(
        is_fraud=bool(pred),
        fraud_probability=round(prob, 4),
        validation_passed=True,
        model_source="MLflow Production",
        model_version=state.model_version,
        request_id=request_id,
    )


@app.get("/health", tags=["Ops"])
def health():
    """Liveness probe. Returns 503 if the model is not loaded."""
    if state.model is None or state.encoder is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or encoder not loaded.",
        )
    return {
        "status": "healthy",
        "validation": "enabled",
        "model_version": state.model_version,
        "loaded_at": state.loaded_at,
    }


@app.get("/model-info", tags=["Ops"])
def model_info():
    """Returns metadata about the currently loaded model."""
    return {
        "registry":      "MLflow",
        "model_name":    "fraud-detection-model",
        "alias":         "champion",
        "model_version": state.model_version,
        "tracking_uri":  mlflow.get_tracking_uri(),
        "loaded_at":     state.loaded_at,
        "valid_categories": sorted(state.valid_categories),
    }