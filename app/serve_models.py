import time
import uuid
import logging
import pickle
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks, status
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
from starlette.responses import Response
from pydantic import BaseModel, Field
from app.config import settings

from app.data_validation import validate_transaction
from app.feast_feature import get_online_features, get_store
from app.db import init_pool, close_pool, persist_prediction, get_drift_history, persist_drift_result, get_recent_predictions
from app.monitoring import DriftMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("fraud_detection")

@dataclass
class AppState:
    model:             object             = None
    encoder:           object             = None
    valid_categories:  set[str]           = field(default_factory=set)
    model_version:     str                = "unknown"
    loaded_at:         float              = 0.0
    drift_monitor:     DriftMonitor|None  = None

FRAUD_THRESHOLD = settings.FRAUD_THRESHOLD
API_KEY = settings.FRAUD_API_KEY
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

async def verify_api_key(
    request: Request,
    key: str | None = Depends(api_key_header),
) -> str:
    """Reusable dependency — raises 401/403 on bad/missing key."""
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server auth not configured.")
    if key is None:
        logger.warning("Missing API key [request_id=%s]", getattr(request.state, "request_id", "unknown"))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    if key != API_KEY:
        logger.warning("Invalid API key [request_id=%s]", getattr(request.state, "request_id", "unknown"))
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key.")
    return key

def get_api_key_for_limiting(request: Request) -> str:
    key = request.headers.get("x-api-key")
    if key:
        import hashlib
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    return get_remote_address(request)

limiter = Limiter(
    key_func=get_api_key_for_limiting,
    default_limits=[settings.RATE_LIMIT_DEFAULT],
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ctx = AppState()
    ctx = app.state.ctx

    t0 = time.perf_counter()
    logger.info("Starting up fraud-detection service …")

    # missing = [v for v in ("FRAUD_API_KEY", "DATABASE_URL") if not os.getenv(v)]
    # if missing:
    #     raise RuntimeError(f"Missing required environment variables: {missing}")
    
    tracking_uri = settings.MLFLOW_TRACKING_URI
    model_uri = f"models:/{settings.MLFLOW_MODEL_NAME}@{settings.MLFLOW_MODEL_ALIAS}"
    mlflow.set_tracking_uri(tracking_uri)
    logger.info("MLflow tracking URI: %s", tracking_uri)

    try:
        ctx.model = mlflow.sklearn.load_model(model_uri)
        logger.info("Champion model loaded from MLflow registry (%s)", model_uri)
    except Exception:
        logger.exception("Failed to load model from MLflow — aborting startup")
        raise

    try:
        client = mlflow.MlflowClient()
        mv = client.get_model_version_by_alias("fraud-detection-model", "champion")
        ctx.model_version = mv.version

        run = client.get_run(mv.run_id)
        optimal_thr = run.data.params.get("optimal_threshold")

        global FRAUD_THRESHOLD
        if optimal_thr is not None:
            FRAUD_THRESHOLD = float(optimal_thr)
            logger.info(
                "Loaded optimal_threshold=%.3f from MLflow run %s",
                FRAUD_THRESHOLD,
                mv.run_id,
            )
        else:
            logger.warning(
                "No optimal_threshold param on run %s — using FRAUD_THRESHOLD=%s",
                mv.run_id,
                FRAUD_THRESHOLD,
            )

        logger.info("Model version: %s (run_id=%s)", mv.version, mv.run_id)
    except Exception:
        logger.warning("Could not fetch model version metadata — continuing anyway")

    encoder_path = "models/encoder.pkl"
    try:
        with open(encoder_path, "rb") as f:
            ctx.encoder = pickle.load(f)
        ctx.valid_categories = set(ctx.encoder.classes_)
        logger.info(
            "Encoder loaded from %s (%d categories)",
            encoder_path,
            len(ctx.valid_categories),
        )
    except FileNotFoundError:
        logger.exception("Encoder file not found at %s", encoder_path)
        raise
    except Exception:
        logger.exception("Failed to load encoder")
        raise

    try:
        get_store()
        logger.info("Feast FeatureStore initialized successfully")
    except Exception:
        logger.exception("Failed to initialize Feast FeatureStore — aborting startup")
        raise

    try:
        reference_path = settings.REFERENCE_DATA_PATH
        ref_df = pd.read_csv(reference_path)
        ctx.drift_monitor = DriftMonitor(
            reference_data=ref_df,
            feature_columns=["amount", "hour", "day_of_week"],
        )
        logger.info("DriftMonitor initialized with %d reference rows", len(ref_df))
    except Exception:
        logger.warning("Could not initialize DriftMonitor — drift checks will be skipped")
        ctx.drift_monitor = None

    try:
        init_pool(min_conn=settings.DB_MIN_CONN, max_conn=settings.DB_MAX_CONN)
        logger.info("DB pool initialized (min=%d, max=%d)", settings.DB_MIN_CONN, settings.DB_MAX_CONN)
    except Exception:
        logger.exception("Failed to initialize DB pool — aborting startup")
        raise

    ctx.loaded_at = time.time()
    logger.info("Startup complete in %.3fs", time.perf_counter() - t0)

    yield

    logger.info("Shutting down fraud-detection service …")
    ctx.model   = None
    ctx.encoder = None
    close_pool()
    logger.info("Resources released. Goodbye.")

app = FastAPI(
    title="Fraud Detection API",
    version="3.0.0",
    description="Real-time transaction fraud scoring powered by MLflow.",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Request-ID + latency middleware
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    t0 = time.perf_counter()

    logger.info("REQUEST  id=%s method=%s path=%s", request_id, request.method, request.url.path)

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

def get_ctx(request: Request) -> AppState:
    return request.app.state.ctx

# Schemas

class Transaction(BaseModel):
    amount: float = Field(..., description="Transaction amount (must be positive)", json_schema_extra={"example": 150.00})
    hour: int = Field(..., description="Hour of day (0-23)", json_schema_extra={"example": 14})
    day_of_week: int = Field(..., description="Day of week (0=Mon, 6=Sun)", json_schema_extra={"example": 3})
    merchant_category: str = Field(..., description="Merchant type", json_schema_extra={"example": "online"})


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    decision_threshold: float = FRAUD_THRESHOLD
    model_source: str = "MLflow Production"
    model_version: str
    validation_passed: bool = True
    feast_status: str = "live"
    request_id: str


class ValidationErrorResponse(BaseModel):
    detail: dict

# Routes

@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ValidationErrorResponse},
        401: {"description": "Missing API key"},
        403: {"description": "Invalid API key"},
        429: {"description": "Rate limit exceeded"},
    },
    tags=["Prediction"],
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(settings.RATE_LIMIT_PREDICT)
def predict(
    tx: Transaction,
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    ctx: AppState = Depends(get_ctx),
):
    """
    Score a transaction for fraud.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    data = tx.model_dump()
    logger.info("Prediction request [request_id=%s] payload=%s", request_id, data)

    # Validation
    validation = validate_transaction(data, ctx.valid_categories)
    if not validation["valid"]:
        logger.warning("Validation failed [request_id=%s] errors=%s", request_id, validation["errors"])
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Validation failed", "errors": validation["errors"], "input": data},
        )

    # Feast features
    feast_features, feast_ok = get_online_features(data["merchant_category"])
    if not feast_ok:
        logger.warning("Feast fallback [request_id=%s] merchant=%r", request_id, data["merchant_category"])

    try:
        data["merchant_encoded"] = ctx.encoder.transform([data["merchant_category"]])[0]
    except Exception:
        logger.exception("Encoder transform failed [request_id=%s]", request_id)
        raise HTTPException(status_code=500, detail="Feature encoding error.")

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
        prob = float(ctx.model.predict_proba(X)[0][1])
        is_fraud = prob >= FRAUD_THRESHOLD
    except Exception:
        logger.exception("Model inference failed [request_id=%s]", request_id)
        raise HTTPException(status_code=500, detail="Model inference error.")

    logger.info(
        "Prediction [request_id=%s] is_fraud=%s probability=%.4f threshold=%.2f",
        request_id,
        is_fraud,
        prob,
        FRAUD_THRESHOLD,
    )

    background_tasks.add_task(
        persist_prediction,
        request_id=request_id,
        amount=data["amount"],
        hour=data["hour"],
        day_of_week=data["day_of_week"],
        merchant_category=data["merchant_category"],
        feast_features=feast_features,
        feast_status="live" if feast_ok else "fallback",
        is_fraud=is_fraud,
        fraud_probability=round(prob, 4),
        model_version=ctx.model_version,
        model_source="MLflow Production",
    )

    return PredictionResponse(
        is_fraud=is_fraud,
        fraud_probability=round(prob, 4),
        decision_threshold=FRAUD_THRESHOLD,
        validation_passed=True,
        feast_status="live" if feast_ok else "fallback",
        model_source="MLflow Production",
        model_version=ctx.model_version,
        request_id=request_id,
    )


@app.get("/monitoring/drift", tags=["Monitoring"])
def drift_summary(ctx: AppState = Depends(get_ctx)):
    """Returns drift check history and summary stats."""
    if ctx.drift_monitor is None:
        raise HTTPException(status_code=503, detail="Drift monitor not available.")

    return {
        "summary": ctx.drift_monitor.summary(),
        "alerts": ctx.drift_monitor.get_alerts(),
        "last_checked_at": ctx.drift_monitor.last_checked_at,
        "history": get_drift_history(limit=50),
    }


@app.get("/monitoring/drift/check", tags=["Monitoring"])
def run_drift_check(
    ctx: AppState = Depends(get_ctx),
    window: int = 100,
):
    """
    Run a batch drift check against the last `window` rows from the DB.
    Useful for scheduled or manual checks.
    """
    if ctx.drift_monitor is None:
        raise HTTPException(status_code=503, detail="Drift monitor not available.")

    rows = get_recent_predictions(limit=window)
    if not rows:
        return {"message": "No predictions in DB yet — cannot run drift check."}

    current_df = pd.DataFrame(rows)
    result = ctx.drift_monitor.check_drift(current_df, window_size=window)

    try:
        persist_drift_result(result=result, window_size=window)
    except Exception:
        logger.exception("Failed to persist drift result for window=%d", window)

    return result


@app.get("/monitoring/stats", tags=["Monitoring"])
def prediction_stats(ctx: AppState = Depends(get_ctx)):
    """Returns live prediction stats from the DB."""
    from app.db import get_prediction_stats
    return get_prediction_stats()


@app.get("/monitoring/recent", tags=["Monitoring"])
def recent_predictions(
    ctx: AppState = Depends(get_ctx),
    limit: int = 100,
):
    """Returns the most recent `limit` predictions from the DB."""
    from app.db import get_recent_predictions
    return get_recent_predictions(limit=limit)


@app.get("/health", tags=["Ops"])
@limiter.exempt
def health(ctx: AppState = Depends(get_ctx)):
    if ctx.model is None or ctx.encoder is None:
        raise HTTPException(status_code=503, detail="Model or encoder not loaded.")
    return {
        "status":           "healthy",
        "model_version":    ctx.model_version,
        "loaded_at":        ctx.loaded_at,
        "drift_monitor":    ctx.drift_monitor is not None,
    }


@app.get("/model-info", tags=["Ops"], dependencies=[Depends(verify_api_key)])
def model_info(ctx: AppState = Depends(get_ctx)):
    return {
        "registry":          "MLflow",
        "model_name":        settings.MLFLOW_MODEL_NAME,
        "alias":             settings.MLFLOW_MODEL_ALIAS,
        "model_version":     ctx.model_version,
        "tracking_uri":      mlflow.get_tracking_uri(),
        "loaded_at":         ctx.loaded_at,
        "valid_categories":  sorted(ctx.valid_categories),
        "decision_threshold": FRAUD_THRESHOLD,
        "env":               settings.ENV,
    }