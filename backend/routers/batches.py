from fastapi import APIRouter, HTTPException
from data_loader import (
    load_production_data, load_process_data,
    get_all_batch_ids, get_batch_summary
)
import pandas as pd

router = APIRouter(prefix="/api/batches", tags=["batches"])

_batches_cache = None


@router.get("")
def list_batches():
    """Return summary for all 60 batches (cached after first call)."""
    global _batches_cache
    if _batches_cache is not None:
        return _batches_cache

    batch_ids = get_all_batch_ids()
    summaries = []
    for bid in batch_ids:
        try:
            s = get_batch_summary(bid)
            summaries.append(s)
        except Exception as e:
            summaries.append({"Batch_ID": bid, "error": str(e)})
    _batches_cache = summaries
    return summaries


@router.get("/{batch_id}")
def get_batch(batch_id: str):
    """Return full time-series data for one batch."""
    ts = load_process_data(batch_id)
    if ts.empty:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
    return {
        "batch_id": batch_id,
        "time_series": ts.to_dict(orient="records"),
        "summary": get_batch_summary(batch_id),
    }


@router.get("/{batch_id}/phases")
def get_batch_phases(batch_id: str):
    """Return per-phase statistics for a batch."""
    ts = load_process_data(batch_id)
    if ts.empty:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")

    phases = []
    for phase, group in ts.groupby("Phase", sort=False):
        phases.append({
            "phase": phase,
            "start_minute": int(group["Time_Minutes"].min()),
            "end_minute": int(group["Time_Minutes"].max()),
            "duration_minutes": len(group),
            "avg_power_kw": round(float(group["Power_Consumption_kW"].mean()), 3),
            "max_power_kw": round(float(group["Power_Consumption_kW"].max()), 3),
            "avg_temperature_c": round(float(group["Temperature_C"].mean()), 2),
            "avg_vibration": round(float(group["Vibration_mm_s"].mean()), 4),
            "avg_pressure_bar": round(float(group["Pressure_Bar"].mean()), 3),
        })
    return phases


@router.get("/ids/all")
def get_batch_ids():
    return get_all_batch_ids()
