import io
import base64
import logging
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image
import pydicom

from app.config import MAX_UPLOAD_SIZE_BYTES

logger = logging.getLogger("spine_health.dicom_service")

class DicomService:
    @staticmethod
    def process_image_file(file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Process DICOM or standard image file entirely in-memory and return base64 PNG with calibration metadata."""
        if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
            raise ValueError(f"File size exceeds maximum allowable limit of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB.")

        lower_name = filename.lower()
        is_dicom = lower_name.endswith(".dcm")

        # Fallback detection: try reading as DICOM even if extension is missing/generic, otherwise standard image
        if is_dicom:
            return DicomService._process_dicom(file_bytes)
        
        # Check if user uploaded a standard image (.png, .jpg, .jpeg, .webp, .bmp)
        supported_img_exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
        if any(lower_name.endswith(ext) for ext in supported_img_exts):
            return DicomService._process_standard_image(file_bytes)

        # If extension isn't recognized, try DICOM first, then standard image
        try:
            return DicomService._process_dicom(file_bytes)
        except Exception:
            try:
                return DicomService._process_standard_image(file_bytes)
            except Exception:
                raise ValueError("Unsupported or corrupted file format. Please upload a valid DICOM (.dcm) or image (.png, .jpg, .jpeg) file.")

    @staticmethod
    def _process_dicom(file_bytes: bytes) -> Dict[str, Any]:
        try:
            with io.BytesIO(file_bytes) as bio:
                ds = pydicom.dcmread(bio, force=True)
            
            if not hasattr(ds, "pixel_array"):
                raise ValueError("DICOM file does not contain a valid pixel array.")

            pixel_array = ds.pixel_array.astype(np.float32)

            # Check PhotometricInterpretation (MONOCHROME1 means 0 is white, needs inversion)
            photometric = getattr(ds, "PhotometricInterpretation", "")
            if photometric == "MONOCHROME1":
                pixel_array = np.max(pixel_array) - pixel_array

            # Fast Min-Max normalization to uint8 (0-255)
            p_min, p_max = pixel_array.min(), pixel_array.max()
            if p_max > p_min:
                normalized_pixels = ((pixel_array - p_min) / (p_max - p_min) * 255.0).astype(np.uint8)
            else:
                normalized_pixels = np.zeros_like(pixel_array, dtype=np.uint8)

            img = Image.fromarray(normalized_pixels)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Extract Pixel Spacing (mm/pixel)
            pixel_spacing_mm = 0.5  # Fallback default
            pixel_spacing = getattr(ds, "PixelSpacing", None) or getattr(ds, "ImagerPixelSpacing", None)
            if pixel_spacing:
                try:
                    pixel_spacing_mm = float(pixel_spacing[0])
                except (IndexError, TypeError, ValueError):
                    pixel_spacing_mm = 0.5

            buffered = io.BytesIO()
            img.save(buffered, format="PNG", optimize=True)
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            return {
                "image_base64": f"data:image/png;base64,{img_b64}",
                "width": img.width,
                "height": img.height,
                "pixel_spacing_mm": round(pixel_spacing_mm, 4),
                "is_dicom": True,
            }
        except Exception as e:
            logger.warning(f"Failed to process DICOM: {e}")
            raise ValueError(f"Corrupted or invalid DICOM file: {str(e)}")

    @staticmethod
    def _process_standard_image(file_bytes: bytes) -> Dict[str, Any]:
        try:
            with io.BytesIO(file_bytes) as bio:
                img = Image.open(bio)
                img.load()
                if img.mode != "RGB":
                    img = img.convert("RGB")

                buffered = io.BytesIO()
                img.save(buffered, format="PNG", optimize=True)
                img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                return {
                    "image_base64": f"data:image/png;base64,{img_b64}",
                    "width": img.width,
                    "height": img.height,
                    "pixel_spacing_mm": 0.5,
                    "is_dicom": False,
                }
        except Exception as e:
            logger.warning(f"Failed to process standard image: {e}")
            raise ValueError(f"Corrupted or invalid image file: {str(e)}")

dicom_service = DicomService()
