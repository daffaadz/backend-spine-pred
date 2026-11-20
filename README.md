# Spine Health Prediction - Backend API

FastAPI backend service untuk inferensi model Machine Learning klasifikasi biomekanik tulang belakang (SVM-ADASYN) serta pemrosesan citra medis lateral X-Ray dan DICOM secara in-memory.

## Fitur Utama

- In-Memory Model: Model SVM Pipeline (StandardScaler + ADASYN + SVC) dimuat saat startup server melalui lifespan event.
- Pemrosesan In-Memory: Konversi citra medis (.dcm, .png, .jpg) dan file CSV diproses langsung dalam memori (io.BytesIO) tanpa penulisan ke disk.
- Performa: Dioptimalkan untuk 10 concurrent requests dengan throughput lebih dari 170 req/s dan latensi P95 di bawah 85 ms.
- CORS: Terintegrasi dengan frontend lokal pada http://localhost:5173.

## Instalasi dan Setup

### 1. Masuk ke Direktori

```bash
cd be-spine-health
```

### 2. Buat dan Aktifkan Virtual Environment

Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Windows (CMD):
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalasi Dependensi

```bash
pip install -r requirements.txt
```

### 4. Menjalankan Server

```bash
python run.py
```

Atau menggunakan uvicorn langsung:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Server aktif pada: http://localhost:8000  
Dokumentasi interaktif OpenAPI (Swagger): http://localhost:8000/docs

## Endpoint API

| Method | Endpoint | Deskripsi |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Status kesehatan server dan ketersediaan model ML |
| `POST` | `/api/predict` | Inferensi tunggal dari 6 parameter spinopelvis |
| `POST` | `/api/predict/batch` | Inferensi batch dari file CSV multi-subjek |
| `POST` | `/api/xray/process` | Normalisasi citra X-Ray / DICOM (.dcm, .png, .jpg) ke Base64 PNG |

### Format Parameter Spinopelvis (POST `/api/predict`)

Input JSON:
```json
{
  "pelvic_incidence": 52.4,
  "pelvic_tilt": 14.2,
  "lumbar_lordosis_angle": 48.1,
  "sacral_slope": 38.2,
  "pelvic_radius": 118.5,
  "degree_spondylolisthesis": 4.3
}
```

Format Output:
```json
{
  "prediction": "Normal",
  "probabilities": {
    "Normal": 0.85,
    "Hernia": 0.10,
    "Spondylolisthesis": 0.05
  },
  "confidence": 0.85,
  "clinical_flags": {
    "pelvic_incidence": "Normal",
    "pelvic_tilt": "Normal",
    "lumbar_lordosis_angle": "Normal",
    "sacral_slope": "Normal",
    "pelvic_radius": "Normal",
    "degree_spondylolisthesis": "Normal"
  }
}
```

## Pengujian dan Verifikasi

Jalankan test suite untuk verifikasi endpoint dan benchmark konkurensi:

```bash
# Unit test API lengkap
python tests/test_api.py

# Benchmark konkurensi (10 concurrent requests)
python tests/benchmark_concurrency.py
```

## Struktur Direktori

```text
be-spine-health/
├── app/
│   ├── main.py              # Inisialisasi FastAPI, CORS, dan lifespan
│   ├── config.py            # Konfigurasi aplikasi dan path model
│   ├── schemas.py           # Pydantic models untuk request dan response
│   ├── routers/
│   │   ├── predict.py       # Endpoint inferensi single dan batch
│   │   └── xray.py          # Endpoint upload dan proses citra X-Ray
│   └── services/
│       ├── ml_service.py    # Logika inferensi SVM dan evaluasi klinis
│       └── dicom_service.py # Handler pemrosesan citra DICOM dan raster
├── models/
│   └── spine_health_svm_adasyn.model
├── tests/
│   ├── test_api.py
│   └── benchmark_concurrency.py
├── requirements.txt
├── run.py
└── README.md
```
