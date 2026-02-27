import numpy as np
import pandas as pd
from data_loader import load_process_data, get_all_batch_ids, INDIA_EMISSION_FACTOR

_carbon_cache = None
_monthly_budget = 500.0  # kg CO2e default monthly budget


def _compute_all_carbon() -> pd.DataFrame:
    global _carbon_cache
    if _carbon_cache is not None:
        return _carbon_cache

    rows = []
    for bid in get_all_batch_ids():
        ts = load_process_data(bid)
        if ts.empty:
            rows.append({"Batch_ID": bid, "total_kwh": 0, "carbon_kg": 0, "duration_min": 0})
            continue
        duration_h = len(ts) / 60.0
        avg_power = ts["Power_Consumption_kW"].mean()
        total_kwh = avg_power * duration_h
        carbon_kg = total_kwh * INDIA_EMISSION_FACTOR
        rows.append({
            "Batch_ID": bid,
            "total_kwh": round(total_kwh, 2),
            "carbon_kg": round(carbon_kg, 2),
            "duration_min": len(ts),
            "avg_power_kw": round(avg_power, 2),
        })

    _carbon_cache = pd.DataFrame(rows)
    return _carbon_cache


def get_per_batch_carbon() -> list:
    df = _compute_all_carbon()
    return df.to_dict(orient="records")


def get_carbon_summary() -> dict:
    global _monthly_budget
    df = _compute_all_carbon()

    total_carbon = df["carbon_kg"].sum()
    total_kwh = df["total_kwh"].sum()
    avg_per_batch = df["carbon_kg"].mean()
    most_efficient = df.loc[df["carbon_kg"].idxmin()]
    least_efficient = df.loc[df["carbon_kg"].idxmax()]

    budget_used_pct = (total_carbon / _monthly_budget * 100) if _monthly_budget else 0
    remaining = max(0, _monthly_budget - total_carbon)

    # Savings opportunities: batches above average that could be optimized
    above_avg = df[df["carbon_kg"] > avg_per_batch]
    potential_savings = (above_avg["carbon_kg"] - avg_per_batch).sum()

    return {
        "total_carbon_kg": round(float(total_carbon), 2),
        "total_kwh": round(float(total_kwh), 2),
        "avg_carbon_per_batch_kg": round(float(avg_per_batch), 2),
        "monthly_budget_kg": _monthly_budget,
        "budget_used_pct": round(float(budget_used_pct), 1),
        "budget_remaining_kg": round(float(remaining), 2),
        "status": "on_track" if budget_used_pct <= 80 else "at_risk" if budget_used_pct <= 100 else "exceeded",
        "most_efficient_batch": most_efficient["Batch_ID"],
        "least_efficient_batch": least_efficient["Batch_ID"],
        "potential_savings_kg": round(float(potential_savings), 2),
        "n_batches": len(df),
    }


def set_monthly_budget(budget_kg: float) -> dict:
    global _monthly_budget
    _monthly_budget = budget_kg
    return {"monthly_budget_kg": _monthly_budget, "status": "updated"}


def get_carbon_forecast() -> dict:
    global _monthly_budget
    df = _compute_all_carbon()

    total_carbon = df["carbon_kg"].sum()
    n_batches = len(df)
    avg_per_batch = df["carbon_kg"].mean()

    # Assume 60 batches/month is typical; forecast for remainder
    assumed_monthly_batches = 60
    batches_remaining = max(0, assumed_monthly_batches - n_batches)
    projected_additional = batches_remaining * avg_per_batch
    projected_total = total_carbon + projected_additional
    projected_vs_budget = projected_total - _monthly_budget

    # Linear trend: is consumption accelerating or decelerating?
    if n_batches >= 10:
        x = np.arange(n_batches)
        y = df["carbon_kg"].values
        coeffs = np.polyfit(x, y, 1)
        trend_direction = "increasing" if coeffs[0] > 0.05 else "decreasing" if coeffs[0] < -0.05 else "stable"
    else:
        trend_direction = "stable"

    return {
        "current_total_kg": round(float(total_carbon), 2),
        "projected_month_end_kg": round(float(projected_total), 2),
        "monthly_budget_kg": _monthly_budget,
        "projected_surplus_deficit_kg": round(float(projected_vs_budget), 2),
        "status": "on_track" if projected_total <= _monthly_budget else "at_risk",
        "trend": trend_direction,
        "batches_processed": n_batches,
        "batches_remaining_estimate": batches_remaining,
        "avg_per_batch_kg": round(float(avg_per_batch), 2),
    }
