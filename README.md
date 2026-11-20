# Spine Health Biomechanical Analyzer — Backend API

Backend REST API berbasis **FastAPI** berkinerja tinggi (*performance-first*) yang melayani inferensi model Machine Learning spinal health classification (SVM-ADASYN) dan pemrosesan citra medis X-Ray/DICOM secara in-memory.

---

## ⚡ Karakteristik & Arsitektur
- **Model ML:** Pre-trained SVM Pipeline (`StandardScaler` + `ADASYN` + `SVC(probability=True)`) dimuat dalam memori saat server startup via FastAPI `lifespan`.
- **Target Concurrency:** Dioptimalkan untuk 10 concurrent users (Throughput teruji > 170 req/s, P95 latensi < 81 ms).
- **Stateless & In-Memory:** Pemrosesan berkas citra dan file batch CSV 100% menggunakan `io.BytesIO` tanpa menyentuh disk fisik.
- **CORS Enabled:** Siap terhubung langsung dengan frontend React/Vite di `http://localhost:5173`.

---

## 🚀 Panduan Menjalankan Backend

### 1. Masuk ke Direktori Backend
```bash
cd be-spine-health
```

### 2. Mengaktifkan Virtual Environment
- **Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Windows (CMD):**
  ```cmd
  venv\Scripts\activate.bat
  ```

### 3. Menjalankan Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Server akan aktif di: `http://localhost:8000`  
Dokumentasi interaktif OpenAPI (Swagger): `http://localhost:8000/docs`

---

## 📡 Ringkasan Endpoint API

| Method | Endpoint | Fungsi |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Pemeriksaan kesehatan server & status pemuatan model ML |
| `POST` | `/api/predict` | Prediksi inferensi tunggal (6 parameter biomekanik) |
| `POST` | `/api/predict/batch` | Prediksi batch berkecepatan tinggi dari berkas CSV |
| `POST` | `/api/xray/process` | Konversi & normalisasi in-memory berkas `.dcm`, `.png`, `.jpg` menjadi base64 PNG |

---

## 🧪 Pengujian & Verifikasi

Jalankan test suite untuk memastikan semua fungsi API dan pipeline inferensi berjalan sempurna:
```bash
# Jalankan unit test lengkap (Health, Single, Batch, CSV Error, DICOM & Standard Image)
python tests/test_api.py

# Jalankan benchmark konkurensi (10 concurrent users)
python tests/benchmark_concurrency.py
```
