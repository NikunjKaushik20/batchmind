from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from models.fingerprint import (
    compute_fingerprints, compute_anomaly_score,
    get_all_anomaly_scores, detect_changepoints,
    compute_degradation_trend, energy_pattern_forecast,
)

router = APIRouter(prefix="/api/fingerprints", tags=["fingerprints"])


@router.get("")
def get_fingerprints():
    """Return phase fingerprint library (DBA barycenter + std band)."""
    return compute_fingerprints()


@router.get("/anomaly/{batch_id}")
def get_anomaly_score(batch_id: str):
    """DTW + Isolation Forest + permutation p-value anomaly report."""
    try:
        return compute_anomaly_score(batch_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/all")
def get_all_health():
    """Asset health scores for all batches."""
    return get_all_anomaly_scores()


@router.get("/changepoints/{batch_id}")
def get_changepoints(batch_id: str):
    """Change-point detection (Ruptures PELT) for phase segmentation validation."""
    try:
        return detect_changepoints(batch_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── ENERGY PATTERN INTELLIGENCE ─────────────────────────────────────────────

@router.get("/degradation")
def degradation_trend(phase: Optional[str] = None):
    """Equipment degradation trend analysis with RUL estimation."""
    try:
        return compute_degradation_trend(phase)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/forecast/{phase}")
def forecast(phase: str, n_future: int = 5):
    """Energy pattern forecast for next N batches."""
    try:
        return energy_pattern_forecast(phase, n_future)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

