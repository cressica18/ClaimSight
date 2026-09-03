"""
ClaimSight backend — application entry point.

Start with:
    uvicorn app.main:app --reload --port 8000
"""

import os
import sys
from pathlib import Path

# Ensure the project root (containing the `ml` package) is on sys.path so the
# CV inference module can be imported when the server is started with the
# documented `uvicorn app.main:app` command from inside the `backend/` directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.health import router as health_router
from app.core.config import settings

app = FastAPI(
    title="ClaimSight API",
    description="AI-Assisted Vehicle Insurance Claims Investigation Platform",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Static Files (uploads) ───────────────────────────────────────────────────
# Serve uploaded images so the frontend can display them
upload_base = Path(settings.upload_dir)
if not upload_base.is_absolute():
    upload_base = Path(os.getcwd()) / upload_base
upload_base.mkdir(parents=True, exist_ok=True)
app.mount("/api/uploads", StaticFiles(directory=str(upload_base)), name="uploads")

# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(health_router, tags=["health"])

from app.api import customers, claims, documents, images, cv, policies, vehicles, pipeline
app.include_router(customers.router, prefix="/customers", tags=["customers"])
app.include_router(policies.router, prefix="/policies", tags=["policies"])
app.include_router(vehicles.router, prefix="/vehicles", tags=["vehicles"])
app.include_router(claims.router, prefix="/claims", tags=["claims"])
app.include_router(documents.router, prefix="/claims", tags=["documents"])
app.include_router(images.router, prefix="/claims", tags=["images"])
app.include_router(cv.router, prefix="/claims", tags=["cv"])
app.include_router(pipeline.router, prefix="/claims", tags=["analysis"])
