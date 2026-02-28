import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import time
import logging

load_dotenv()

logger = logging.getLogger(__name__)

from routers import batches, fingerprint, optimizer, golden, carbon
from routers import copilot

app = FastAPI(
    title="BatchMind API",
    description="AI-Driven Manufacturing Intelligence — Causal-Physics Hybrid System",
    version="1.0.0",
)

# ─── CORS (configurable origins from env) ────────────────────────────────────

ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── REQUEST TIMING MIDDLEWARE ────────────────────────────────────────────────

@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Add X-Process-Time header to all responses for performance monitoring."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{elapsed:.3f}s"
    if elapsed > 5.0:
        logger.warning(f"Slow request: {request.method} {request.url.path} took {elapsed:.1f}s")
    return response


# ─── GLOBAL EXCEPTION HANDLER ────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions and return structured error responses."""
    logger.error(f"Unhandled error on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ─── ROUTERS ─────────────────────────────────────────────────────────────────

app.include_router(batches.router)
app.include_router(fingerprint.router)
app.include_router(optimizer.router)
app.include_router(golden.router)
app.include_router(carbon.router)
app.include_router(copilot.router)


@app.get("/")
def root():
    return {
        "name": "BatchMind API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
        "copilot": "/api/copilot/chat",
    }


@app.get("/health")
def health():
    return {"status": "ok"}
