from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from models.fingerprint import (
    compute_fingerprints, compute_anomaly_score,
    get_all_anomaly_scores, detect_changepoints,
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
