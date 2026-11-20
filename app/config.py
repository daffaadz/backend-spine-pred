import os
from pathlib import Path
from typing import Dict

BASE_DIR = Path(__file__).resolve().parent.parent

# Model paths
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "spine_health_svm_adasyn.model"))
FALLBACK_MODEL_PATH = BASE_DIR / "models" / "model_spine.pkl"

MOCK_MODEL = os.getenv("MOCK_MODEL", "false").lower() in ("true", "1", "yes")
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_MB", "20")) * 1024 * 1024

# Strict feature order required by model and contracts
FEATURE_NAMES = [
    "pelvic_incidence",
    "pelvic_tilt",
    "lumbar_lordosis_angle",
    "sacral_slope",
    "pelvic_radius",
    "degree_spondylolisthesis",
]

# Clinical ranges definition and helper evaluator
CLINICAL_RANGES = {
    "pelvic_incidence": {"min": 40.0, "max": 65.0, "unit": "°"},
    "pelvic_tilt": {"min": 10.0, "max": 20.0, "unit": "°"},
    "lumbar_lordosis_angle": {"min": 40.0, "max": 60.0, "unit": "°"},
    "sacral_slope": {"min": 35.0, "max": 45.0, "unit": "°"},
    "pelvic_radius": {"min": 110.0, "max": 140.0, "unit": "mm"},
    "degree_spondylolisthesis": {"max": 5.0, "unit": "mm"},
}

CLASS_LABEL_MAP: Dict[str, str] = {
    "NO": "Normal",
    "DH": "Hernia",
    "SL": "Spondylolisthesis",
    "Normal": "Normal",
    "Hernia": "Hernia",
    "Spondylolisthesis": "Spondylolisthesis",
}

def evaluate_clinical_flag(feature: str, value: float) -> str:
    """Evaluate if a biomechanical feature falls within the standard clinical reference range."""
    if feature == "pelvic_incidence":
        if value < 40.0:
            return f"Low ({value:.1f}° < 40°)"
        if value > 65.0:
            return f"High ({value:.1f}° > 65°)"
        return "Normal (40-65°)"
    
    if feature == "pelvic_tilt":
        if value < 10.0:
            return f"Low ({value:.1f}° < 10°)"
        if value > 20.0:
            return f"High ({value:.1f}° > 20°)"
        return "Normal (10-20°)"
    
    if feature == "lumbar_lordosis_angle":
        if value < 40.0:
            return f"Hypolordosis ({value:.1f}° < 40°)"
        if value > 60.0:
            return f"Hyperlordosis ({value:.1f}° > 60°)"
        return "Normal (40-60°)"
    
    if feature == "sacral_slope":
        if value < 35.0:
            return f"Low ({value:.1f}° < 35°)"
        if value > 45.0:
            return f"High ({value:.1f}° > 45°)"
        return "Normal (35-45°)"
    
    if feature == "pelvic_radius":
        if value < 110.0:
            return f"Low ({value:.1f} mm < 110 mm)"
        if value > 140.0:
            return f"High ({value:.1f} mm > 140 mm)"
        return "Normal (110-140 mm)"
    
    if feature == "degree_spondylolisthesis":
        if value >= 5.0:
            return f"Elevated ({value:.1f} mm ≥ 5 mm)"
        return "Normal (<5 mm)"
    
    return "Normal"
