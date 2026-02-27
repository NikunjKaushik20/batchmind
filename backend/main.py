import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from routers import batches, fingerprint, optimizer, golden, carbon
from routers import copilot

app = FastAPI(
    title="BatchMind API",
    description="AI-Driven Manufacturing Intelligence — Causal-Physics Hybrid System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
