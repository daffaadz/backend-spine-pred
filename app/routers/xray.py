import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, status

from app.schemas import XrayProcessResponse
from app.services.dicom_service import dicom_service
from app.config import MAX_UPLOAD_SIZE_BYTES

logger = logging.getLogger("spine_health.router.xray")

router = APIRouter(prefix="/api/xray", tags=["X-Ray / DICOM Imaging"])

@router.post(
    "/process",
    response_model=XrayProcessResponse,
    summary="Process medical X-Ray or DICOM file",
    description="Accepts an uploaded .dcm, .png, or .jpg file, performs in-memory normalization, extracts physical calibration spacing, and returns a browser-renderable base64 PNG string.",
)
async def process_xray(file: UploadFile = File(...)):
    filename = file.filename or "uploaded_image"
    file_bytes = await file.read()

    if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowable size of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB.",
        )

    try:
        result = dicom_service.process_image_file(file_bytes, filename)
        return result
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as e:
        logger.error(f"X-Ray processing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process image: {str(e)}",
        )
