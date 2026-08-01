"""
FastAPI REST API — Real-time prediction endpoints.

Provides:
- GET /health: Health check and model status
- GET /metrics: Basic API usage and performance metrics
- POST /predict: Single customer prediction
- POST /predict/batch: CSV upload for bulk predictions
"""

import time
import logging
import io
import json
import datetime
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from pipeline_lib import FINAL_PIPELINE_PATH, FEATURE_NAMES_PATH, MODEL_METADATA_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
from app.schemas import CustomerInput, PredictionResponse, BatchPredictionResponse, HealthResponse

logger = get_logger("api")

# =============================================================================
# App Initialization & State
# =============================================================================

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="REST API for predicting telecom customer churn.",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
START_TIME = time.time()
metrics = {
    "total_predictions": 0,
    "batch_requests": 0,
    "errors": 0,
    "avg_latency_ms": 0.0,
    "predict_requests": 0,  # request count backing the latency average
}

# Lazy loading of model to avoid startup crashes if model isn't built yet.
# The loaded pipeline is keyed by the model file's mtime, so retraining
# (or artifacts appearing after API startup) is picked up automatically —
# no stale globals, no restart required.
MODEL_PIPELINE = None
FEATURE_NAMES = None
DECISION_THRESHOLD = 0.5  # Overridden by tuned value from model metadata
_LOADED_MODEL_MTIME = None


def load_model():
    """Load the trained model pipeline, feature names, and tuned threshold."""
    global MODEL_PIPELINE, FEATURE_NAMES, DECISION_THRESHOLD, _LOADED_MODEL_MTIME

    try:
        if not FINAL_PIPELINE_PATH.exists():
            logger.warning(f"Model file not found at {FINAL_PIPELINE_PATH}. Run training first.")
            MODEL_PIPELINE = None
            _LOADED_MODEL_MTIME = None
            return None, None

        mtime = FINAL_PIPELINE_PATH.stat().st_mtime
        if MODEL_PIPELINE is not None and mtime == _LOADED_MODEL_MTIME:
            return MODEL_PIPELINE, FEATURE_NAMES

        MODEL_PIPELINE = joblib.load(FINAL_PIPELINE_PATH)
        _LOADED_MODEL_MTIME = mtime

        if FEATURE_NAMES_PATH.exists():
            FEATURE_NAMES = joblib.load(FEATURE_NAMES_PATH)
        else:
            logger.warning("Feature names file not found. Assuming raw input matches training.")

        DECISION_THRESHOLD = 0.5
        if MODEL_METADATA_PATH.exists():
            try:
                metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
                DECISION_THRESHOLD = float(metadata.get("decision_threshold", 0.5))
                logger.info(
                    f"Loaded metadata: model={metadata.get('best_model')}, "
                    f"threshold={DECISION_THRESHOLD:.3f}"
                )
            except (ValueError, json.JSONDecodeError) as e:
                logger.warning(f"Could not parse model metadata: {e}. Using threshold 0.5.")

        logger.info(f"Successfully loaded model from {FINAL_PIPELINE_PATH}")
        return MODEL_PIPELINE, FEATURE_NAMES

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        MODEL_PIPELINE = None
        _LOADED_MODEL_MTIME = None
        return None, None


# =============================================================================
# Middleware
# =============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and update latency metrics."""
    start_time = time.time()
    
    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        
        # Update rolling average latency — averaged over REQUESTS, not
        # prediction rows (a 1000-row batch is still one latency sample)
        if request.url.path.startswith("/predict"):
            n = metrics["predict_requests"]
            metrics["avg_latency_ms"] = (metrics["avg_latency_ms"] * n + process_time) / (n + 1)
            metrics["predict_requests"] = n + 1
        
        logger.info(
            f"{request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Latency: {process_time:.2f}ms"
        )
        return response
        
    except Exception as e:
        metrics["errors"] += 1
        logger.error(f"Unhandled exception during {request.method} {request.url.path}: {str(e)}")
        raise


# =============================================================================
# Helper Functions
# =============================================================================

def determine_risk_level(probability: float) -> str:
    """Map prediction probability to a business risk level."""
    if probability < 0.30:
        return "Low"
    elif probability < 0.65:
        return "Medium"
    else:
        return "High"

def process_prediction(df: pd.DataFrame, pipeline) -> list:
    """Run prediction and return formatted results."""
    # The pipeline is fully self-contained: feature engineering, encoding,
    # and scaling all happen inside predict_proba. Raw records go straight in.
    probabilities = pipeline.predict_proba(df)[:, 1]
    predictions = (probabilities >= DECISION_THRESHOLD).astype(int)
    
    results = []
    for pred, prob in zip(predictions, probabilities):
        results.append(PredictionResponse(
            churn_prediction=int(pred),
            churn_probability=float(prob),
            risk_level=determine_risk_level(prob)
        ))
        
    return results


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/", include_in_schema=False)
def root():
    """Redirect root to Swagger documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health_check():
    """API health check endpoint."""
    pipeline, _ = load_model()
    
    status = "healthy" if pipeline is not None else "degraded (model not loaded)"
    uptime = time.time() - START_TIME
    
    # Try to get modification time of model file for versioning
    model_version = "unknown"
    if FINAL_PIPELINE_PATH.exists():
        mtime = FINAL_PIPELINE_PATH.stat().st_mtime
        model_version = datetime.datetime.fromtimestamp(mtime).isoformat()
        
    return HealthResponse(
        status=status,
        model_version=model_version,
        uptime_seconds=uptime,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


@app.get("/metrics")
def get_metrics():
    """Return basic API metrics."""
    return metrics


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerInput):
    """Predict churn probability for a single customer."""
    pipeline, _ = load_model()
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or available.")
        
    try:
        # Convert single input to DataFrame
        df = pd.DataFrame([customer.model_dump()])
        
        # Run prediction
        result = process_prediction(df, pipeline)[0]
        
        # Update metrics
        metrics["total_predictions"] += 1
        
        return result
        
    except Exception as e:
        metrics["errors"] += 1
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(file: UploadFile = File(...)):
    """Predict churn for a batch of customers uploaded via CSV."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    pipeline, _ = load_model()
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or available.")
        
    try:
        # Read CSV file
        contents = file.file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Run predictions
        results = process_prediction(df, pipeline)
        
        # Update metrics
        metrics["batch_requests"] += 1
        metrics["total_predictions"] += len(results)
        
        return BatchPredictionResponse(
            predictions=results,
            total_processed=len(results)
        )
        
    except Exception as e:
        metrics["errors"] += 1
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}")
