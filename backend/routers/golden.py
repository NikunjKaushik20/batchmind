from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from models.golden_signature import (
    compute_golden_signature, get_all_signatures,
    get_pareto_data, check_and_update_signature,
    get_bayesian_state, bayesian_update_from_feedback,
)

router = APIRouter(prefix="/api/golden", tags=["golden"])


class SignatureRequest(BaseModel):
    quality: float = 0.25
    yield_score: float = 0.25
    energy: float = 0.25
    performance: float = 0.1
    label: str = None


class UpdateRequest(BaseModel):
    sig_id: str
    new_batch_id: str


class BayesianFeedbackRequest(BaseModel):
    sig_id: str
    param_values: dict


@router.get("")
def list_signatures():
    return get_all_signatures()


@router.post("")
def create_signature(req: SignatureRequest):
    objectives = {
        "quality": req.quality, "yield": req.yield_score,
        "energy": req.energy, "performance": req.performance,
    }
    try:
        sig = compute_golden_signature(objectives, label=req.label)
        return sig
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pareto")
def pareto_data():
    """Both historical batch data + NSGA-II Pareto-optimal solutions."""
    return get_pareto_data()


@router.post("/update")
def update_signature(req: UpdateRequest):
    """Check Pareto dominance + Bayesian posterior update."""
    return check_and_update_signature(req.sig_id, req.new_batch_id)


# ─── NEW BAYESIAN ENDPOINTS ──────────────────────────────────────────────────

@router.get("/bayesian-state")
def bayesian_state(sig_id: str = None):
    """
    Return full Bayesian posterior state (NIG parameters, credible intervals).
    This is the mathematical proof that Bayesian updating is real.
    """
    return get_bayesian_state(sig_id)


@router.post("/bayesian-update")
def bayesian_update(req: BayesianFeedbackRequest):
    """
    Manual Bayesian update from operator feedback.
    Operator approves parameter values → NIG posterior updates.
    """
    return bayesian_update_from_feedback(req.sig_id, req.param_values)
