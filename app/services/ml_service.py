import os
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import joblib

from app.config import (
    MODEL_PATH,
    FALLBACK_MODEL_PATH,
    MOCK_MODEL,
    FEATURE_NAMES,
    CLASS_LABEL_MAP,
    evaluate_clinical_flag,
)

logger = logging.getLogger("spine_health.ml_service")

class MLService:
    _instance: Optional["MLService"] = None

    def __init__(self):
        self.model: Any = None
        self.classes: List[str] = []
        self.is_loaded: bool = False
        self.is_mock: bool = False

    @classmethod
    def get_instance(cls) -> "MLService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self) -> None:
        """Load the pre-trained model artifact into memory."""
        target_path = MODEL_PATH if MODEL_PATH.exists() else FALLBACK_MODEL_PATH

        if not target_path.exists():
            if MOCK_MODEL:
                logger.warning(f"Model file not found at {target_path}. Running in MOCK_MODEL mode.")
                self.is_loaded = True
                self.is_mock = True
                self.classes = ["Normal", "Hernia", "Spondylolisthesis"]
                return
            else:
                logger.error(f"Model file not found at {target_path}")
                self.is_loaded = False
                return

        try:
            loaded_artifact = joblib.load(target_path)
            
            # If the artifact is a dictionary bundle (e.g. from notebook training pipeline)
            if isinstance(loaded_artifact, dict) and "model" in loaded_artifact:
                self.model = loaded_artifact["model"]
            else:
                self.model = loaded_artifact

            if hasattr(self.model, "classes_"):
                self.classes = [
                    CLASS_LABEL_MAP.get(str(c), str(c)) for c in self.model.classes_
                ]
            else:
                self.classes = ["Normal", "Hernia", "Spondylolisthesis"]

            self.is_loaded = True
            self.is_mock = False
            logger.info(f"Model successfully loaded from {target_path}. Classes: {self.classes}")
        except Exception as e:
            logger.exception(f"Failed to load model from {target_path}: {e}")
            if MOCK_MODEL:
                logger.warning("Falling back to MOCK_MODEL mode due to load error.")
                self.is_loaded = True
                self.is_mock = True
                self.classes = ["Normal", "Hernia", "Spondylolisthesis"]
            else:
                self.is_loaded = False
                raise e

    def predict_single(self, features_dict: Dict[str, float]) -> Dict[str, Any]:
        """Perform high-performance single inference and generate clinical reference flags."""
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Ensure service startup succeeded.")

        # Wrap in DataFrame with feature names to keep StandardScaler fitted column names
        x_df = pd.DataFrame([features_dict], columns=FEATURE_NAMES)

        if self.is_mock:
            prediction = "Normal"
            probabilities = {"Normal": 0.85, "Hernia": 0.10, "Spondylolisthesis": 0.05}
        else:
            raw_pred = self.model.predict(x_df)[0]
            prediction = CLASS_LABEL_MAP.get(str(raw_pred), str(raw_pred))

            if hasattr(self.model, "predict_proba"):
                raw_proba = self.model.predict_proba(x_df)[0]
                probabilities = {}
                for idx, cls_raw in enumerate(self.model.classes_):
                    label = CLASS_LABEL_MAP.get(str(cls_raw), str(cls_raw))
                    probabilities[label] = round(float(raw_proba[idx]), 4)
            else:
                probabilities = {prediction: 1.0}

        # Calculate clinical parameter flags
        clinical_flags = {
            feat: evaluate_clinical_flag(feat, features_dict[feat])
            for feat in FEATURE_NAMES
        }

        return {
            "prediction": prediction,
            "probabilities": probabilities,
            "clinical_flags": clinical_flags,
        }

    def predict_batch(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform vectorized batch inference on a pandas DataFrame."""
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Ensure service startup succeeded.")

        # Validate that all required columns are present
        missing_cols = [col for col in FEATURE_NAMES if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {missing_cols}")

        # Check for NaN / non-numeric values in required columns
        df_subset = df[FEATURE_NAMES]
        if df_subset.isna().any().any():
            raise ValueError("CSV contains missing (NaN) values in required biomechanical features.")

        df_subset = df[FEATURE_NAMES].astype(float)

        total_rows = len(df)
        if total_rows == 0:
            return {"total_rows": 0, "summary": {}, "results": []}

        if self.is_mock:
            predictions = ["Normal"] * total_rows
            confidences = [0.85] * total_rows
        else:
            raw_preds = self.model.predict(df_subset)
            predictions = [CLASS_LABEL_MAP.get(str(p), str(p)) for p in raw_preds]

            if hasattr(self.model, "predict_proba"):
                raw_probas = self.model.predict_proba(df_subset)
                confidences = [round(float(np.max(p)), 4) for p in raw_probas]
            else:
                confidences = [1.0] * total_rows

        x_matrix = df_subset.to_numpy()

        # Generate summary counts
        summary: Dict[str, int] = {}
        for pred in predictions:
            summary[pred] = summary.get(pred, 0) + 1

        # Format row items
        results = []
        for idx in range(total_rows):
            row_dict = {
                "row_index": idx,
                "prediction": predictions[idx],
                "confidence": confidences[idx],
            }
            for feat in FEATURE_NAMES:
                row_dict[feat] = round(float(x_matrix[idx, FEATURE_NAMES.index(feat)]), 4)
            results.append(row_dict)

        return {
            "total_rows": total_rows,
            "summary": summary,
            "results": results,
        }

ml_service = MLService.get_instance()
