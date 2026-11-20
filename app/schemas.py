from typing import Dict, List
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool

class XrayProcessResponse(BaseModel):
    image_base64: str
    width: int
    height: int
    pixel_spacing_mm: float
    is_dicom: bool

class PredictRequest(BaseModel):
    pelvic_incidence: float = Field(..., description="Pelvic Incidence in degrees")
    pelvic_tilt: float = Field(..., description="Pelvic Tilt in degrees")
    lumbar_lordosis_angle: float = Field(..., description="Lumbar Lordosis Angle in degrees")
    sacral_slope: float = Field(..., description="Sacral Slope in degrees")
    pelvic_radius: float = Field(..., description="Pelvic Radius in mm")
    degree_spondylolisthesis: float = Field(..., description="Degree of Spondylolisthesis in mm")

class PredictResponse(BaseModel):
    prediction: str
    probabilities: Dict[str, float]
    clinical_flags: Dict[str, str]

class BatchItemResult(BaseModel):
    row_index: int
    pelvic_incidence: float
    pelvic_tilt: float
    lumbar_lordosis_angle: float
    sacral_slope: float
    pelvic_radius: float
    degree_spondylolisthesis: float
    prediction: str
    confidence: float

class BatchPredictResponse(BaseModel):
    total_rows: int
    summary: Dict[str, int]
    results: List[BatchItemResult]

class ErrorDetail(BaseModel):
    detail: str
