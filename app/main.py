import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import HealthResponse
from app.services.ml_service import ml_service
from app.routers import predict, xray

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("spine_health.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ML model into memory once
    logger.info("Initializing Spine Health Analyzer API...")
    try:
        ml_service.load_model()
    except Exception as e:
        logger.error(f"Startup model load failed: {e}")
    yield
    # Shutdown
    logger.info("Shutting down Spine Health Analyzer API...")

app = FastAPI(
    title="Spine Health Biomechanical Analyzer API",
    description="High-performance, stateless REST API for spinal health classification and X-Ray processing.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration allowing local Vite frontend and general development origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount feature routers
app.include_router(predict.router)
app.include_router(xray.router)

@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Health check endpoint indicating API operational status and model readiness."""
    return HealthResponse(
        status="ok",
        model_loaded=ml_service.is_loaded,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
