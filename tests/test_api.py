import io
import time
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

from app.main import app
from app.services.ml_service import ml_service

# Ensure model is loaded for tests
ml_service.load_model()
client = TestClient(app)

def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    print("\n[PASS] Health check verified:", data)

def test_predict_single():
    payload = {
        "pelvic_incidence": 52.4,
        "pelvic_tilt": 14.2,
        "lumbar_lordosis_angle": 48.1,
        "sacral_slope": 38.2,
        "pelvic_radius": 118.5,
        "degree_spondylolisthesis": 4.3,
    }

    t0 = time.perf_counter()
    response = client.post("/api/predict", json=payload)
    latency_ms = (time.perf_counter() - t0) * 1000

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in ["Normal", "Hernia", "Spondylolisthesis"]
    assert "probabilities" in data
    assert "clinical_flags" in data
    print(f"\n[PASS] Single predict completed in {latency_ms:.2f}ms")
    print("Result:", data)

def test_predict_batch():
    csv_content = """pelvic_incidence,pelvic_tilt,lumbar_lordosis_angle,sacral_slope,pelvic_radius,degree_spondylolisthesis
52.4,14.2,48.1,38.2,118.5,4.3
63.02,22.55,39.60,40.47,98.67,-0.25
39.05,10.06,25.01,28.99,114.40,4.56
70.5,25.1,65.0,45.4,105.0,35.8
"""
    files = {"file": ("test_cases.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}

    t0 = time.perf_counter()
    response = client.post("/api/predict/batch", files=files)
    latency_ms = (time.perf_counter() - t0) * 1000

    assert response.status_code == 200
    data = response.json()
    assert data["total_rows"] == 4
    assert len(data["results"]) == 4
    assert "summary" in data
    print(f"\n[PASS] Batch predict 4 rows completed in {latency_ms:.2f}ms")
    print("Summary:", data["summary"])

def test_predict_batch_missing_columns():
    csv_invalid = """pelvic_incidence,pelvic_tilt
52.4,14.2
"""
    files = {"file": ("invalid.csv", io.BytesIO(csv_invalid.encode("utf-8")), "text/csv")}
    response = client.post("/api/predict/batch", files=files)
    assert response.status_code == 422
    print("\n[PASS] Invalid CSV rejected with 422 Unprocessable Entity:", response.json())

def test_xray_process_standard_image():
    # Create a synthetic 100x100 test PNG image
    arr = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    bio.seek(0)

    files = {"file": ("test_xray.png", bio, "image/png")}
    response = client.post("/api/xray/process", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["is_dicom"] is False
    assert data["width"] == 100
    assert data["height"] == 100
    assert data["pixel_spacing_mm"] == 0.5
    assert data["image_base64"].startswith("data:image/png;base64,")
    print("\n[PASS] Standard image X-Ray processing verified:", {k: v for k, v in data.items() if k != "image_base64"})

def test_xray_process_dicom():
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage

    # Create in-memory minimal synthetic DICOM file
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.5"
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    bio = io.BytesIO()
    ds = FileDataset(bio, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.Modality = "OT"
    ds.Rows = 64
    ds.Columns = 64
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelSpacing = ["0.25", "0.25"]
    arr = (np.random.rand(64, 64) * 1000).astype(np.uint16)
    ds.PixelData = arr.tobytes()

    ds.save_as(bio, write_like_original=False)
    bio.seek(0)

    files = {"file": ("test_spine.dcm", bio, "application/dicom")}
    response = client.post("/api/xray/process", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["is_dicom"] is True
    assert data["width"] == 64
    assert data["height"] == 64
    assert data["pixel_spacing_mm"] == 0.25
    assert data["image_base64"].startswith("data:image/png;base64,")
    print("\n[PASS] DICOM X-Ray processing verified:", {k: v for k, v in data.items() if k != "image_base64"})

if __name__ == "__main__":
    test_health()
    test_predict_single()
    test_predict_batch()
    test_predict_batch_missing_columns()
    test_xray_process_standard_image()
    test_xray_process_dicom()
    print("\nALL API TESTS PASSED!")
