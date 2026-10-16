"""FastAPI entry point.  Run from the project root:

    python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
"""
from fastapi import FastAPI

from backend.api_registry import build_router

app = FastAPI(title="HydroMorph-Analytics Compute Engine", version="1.0.0")
app.include_router(build_router())


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"message": "HydroMorph-Analytics Compute Engine", "docs": "/docs"}


@app.get("/favicon.ico")
def favicon():
    return {"detail": "no favicon"}
