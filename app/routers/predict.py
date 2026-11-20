import io
import logging
import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File, status

from app.schemas import (
    PredictRequest,
    PredictResponse,
    BatchPredictResponse,
)
from app.services.ml_service import ml_service
from app.config import MAX_UPLOAD_SIZE_BYTES

logger = logging.getLogger("spine_health.router.predict")

router = APIRouter(prefix="/api/predict", tags=["Prediction"])

@router.post(
    "",
    response_model=PredictResponse,
    summary="Run inference on 6 spinopelvic features",
    description="Accepts 6 biomechanical angles/distances from manual input or landmark calculation, returns predicted pathology, class probabilities, and clinical reference flags.",
)
def predict_single(payload: PredictRequest):
    if not ml_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine Learning model is not loaded. Please verify backend startup status.",
        )
    try:
        result = ml_service.predict_single(payload.model_dump())
        return result
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Inference failed: {str(e)}",
        )

@router.post(
    "/batch",
    response_model=BatchPredictResponse,
    summary="Batch inference from uploaded CSV file",
    description="Processes an uploaded CSV containing the 6 required spinopelvic parameter columns and yields vectorized predictions.",
)
async def predict_batch(file: UploadFile = File(...)):
    if not ml_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine Learning model is not loaded. Please verify backend startup status.",
        )

    filename = file.filename or "unknown.csv"
    if not filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid file type. Only .csv files are supported for batch inference.",
        )

    file_bytes = await file.read()
    if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum upload limit of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB.",
        )

    try:
        # Load CSV in memory
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        logger.warning(f"CSV parse error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse CSV file: {str(e)}",
        )

    try:
        batch_result = ml_service.predict_batch(df)
        return batch_result
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve),
        )
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch prediction error: {str(e)}",
        )
