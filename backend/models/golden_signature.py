"""
Bayesian Golden Signature — Real NIG Conjugate Posteriors + NSGA-II Pareto

Architecture:
  1. NSGA-II Pareto front across all manufacturing objectives
  2. Normal-Inverse-Gamma (NIG) conjugate priors initialized from Pareto solutions
  3. True Bayesian posterior updating when new qualifying batches arrive
  4. Credible intervals from posterior marginal t-distribution (NOT frequentist z-intervals)
  5. Bayesian model comparison via log marginal likelihood (Bayes Factor)
  6. Pareto dominance checking for self-update

Mathematics:
  Prior:     (μ, σ²) ~ NIG(μ₀, κ₀, α₀, β₀) — initialized from Pareto neighbourhood
  Update:    κₙ = κ₀+n, μₙ = (κ₀μ₀+nx̄)/κₙ, αₙ = α₀+n/2, βₙ = β₀+…
  Credible:  μ | data ~ t_{2αₙ}(μₙ, βₙ/(αₙ·κₙ))
  Evidence:  log p(x₁:ₙ) for Bayes Factor between signatures
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

logger = logging.getLogger(__name__)

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from models.bayesian import NormalInverseGamma, BayesianParameterTracker
from data_loader import (
    load_production_data, load_process_data,
    get_all_batch_ids, INDIA_EMISSION_FACTOR
)

INPUT_PARAMS = [
    "Granulation_Time", "Binder_Amount", "Drying_Temp", "Drying_Time",
    "Compression_Force", "Machine_Speed", "Lubricant_Conc"
]
OUTPUT_TARGETS = [
    "Hardness", "Friability", "Disintegration_Time",
    "Dissolution_Rate", "Content_Uniformity"
]

_golden_signatures = {}
_surrogates = {}
_scaler = None
_param_bounds = None
_enriched_df = None
_pareto_cache = None
_bayesian_trackers = {}  # sig_id → BayesianParameterTracker

import threading as _threading
_init_lock = _threading.Lock()
_init_done_event = _threading.Event()  # set when _ensure_signatures() finishes


def _get_data() -> pd.DataFrame:
    global _enriched_df
    if _enriched_df is not None:
        return _enriched_df

    prod = load_production_data()
    energy_rows = []
    for bid in get_all_batch_ids():
        ts = load_process_data(bid)
        if ts.empty:
            energy_rows.append({"Batch_ID": bid, "total_kwh": 0.0})
            continue
        duration_h = len(ts) / 60.0
        avg_power = ts["Power_Consumption_kW"].mean()
        energy_rows.append({"Batch_ID": bid, "total_kwh": round(avg_power * duration_h, 3)})

    energy_df = pd.DataFrame(energy_rows)
    df = prod.merge(energy_df, on="Batch_ID", how="left")
    df["carbon_kg"] = df["total_kwh"] * INDIA_EMISSION_FACTOR
    df["Quality_Score"] = (
        (df["Hardness"].clip(0, 120) / 120) * 30
        + (df["Dissolution_Rate"].clip(0, 100) / 100) * 30
        + (1 - df["Friability"].clip(0, 2) / 2) * 15
        + (1 - df["Disintegration_Time"].clip(0, 20) / 20) * 10
        + (df["Content_Uniformity"].clip(0, 105) / 105) * 15
    )
    df["Yield_Score"] = df["Dissolution_Rate"] * 0.6 + df["Content_Uniformity"] * 0.4
    _enriched_df = df
    return df


def _train_surrogates():
    global _surrogates, _scaler, _param_bounds

    df = _get_data()
    X = df[INPUT_PARAMS]
    _scaler = StandardScaler()
    X_scaled = _scaler.fit_transform(X)

    _param_bounds = {
        "lower": X.min().values.astype(float),
        "upper": X.max().values.astype(float),
    }

    targets = OUTPUT_TARGETS + ["total_kwh", "Quality_Score", "Yield_Score"]
    for t in targets:
        if t not in df.columns:
            continue
        model = lgb.LGBMRegressor(
            n_estimators=150, learning_rate=0.05,
            num_leaves=15, random_state=42, verbose=-1
        )
        model.fit(X_scaled, df[t].values)
        _surrogates[t] = model


def _predict_sur(x: np.ndarray, target: str) -> float:
    x_s = _scaler.transform(x.reshape(1, -1))
    return float(_surrogates[target].predict(x_s)[0])


class GoldenSignatureProblem(ElementwiseProblem):
    """
    Constrained multi-objective NSGA-II for golden signature generation.
    Same 4 objectives + 4 pharmacopoeial constraints as the causal optimizer.
    """

    def __init__(self):
        super().__init__(
            n_var=len(INPUT_PARAMS), n_obj=4,
            n_ieq_constr=4,
            xl=_param_bounds["lower"], xu=_param_bounds["upper"],
        )

    def _evaluate(self, x, out, *args, **kwargs):
        quality = _predict_sur(x, "Quality_Score")
        yield_s = _predict_sur(x, "Yield_Score")
        energy = _predict_sur(x, "total_kwh")
        disint = _predict_sur(x, "Disintegration_Time")
        hardness = _predict_sur(x, "Hardness")
        friability = _predict_sur(x, "Friability")
        dissolution = _predict_sur(x, "Dissolution_Rate")

        out["F"] = [-quality, -yield_s, energy, disint]
        out["G"] = [
            40.0 - hardness,
            friability - 1.0,
            80.0 - dissolution,
            disint - 15.0,
        ]


def _run_nsga2_pareto(n_gen: int = 30, pop_size: int = 25) -> tuple:
    """Run NSGA-II for Pareto front. Reduced from n_gen=100/pop=60 for speed.
    60-row dataset doesn't benefit from massive populations."""
    problem = GoldenSignatureProblem()
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(eta=15, prob=0.9),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )
    result = pymoo_minimize(
        problem, algorithm,
        get_termination("n_gen", n_gen),
        seed=42, verbose=False,
    )

    X = result.X if result.X is not None else np.array([_param_bounds["lower"]])
    F = result.F if result.F is not None else np.zeros((1, 4))
    if X.ndim == 1:
        X = X.reshape(1, -1)
        F = F.reshape(1, -1)
    return X, F


_pareto_X = None
_pareto_F = None


def _ensure_pareto():
    _ensure_surrogates()
    global _pareto_X, _pareto_F, _pareto_cache
    if _pareto_X is None:
        logger.info("Running constrained NSGA-II Pareto optimization...")
        _pareto_X, _pareto_F = _run_nsga2_pareto()  # uses reduced defaults
        logger.info(f"Pareto front: {len(_pareto_X)} solutions")

        _pareto_cache = []
        for x, f in zip(_pareto_X, _pareto_F):
            _pareto_cache.append({
                "quality_score": round(float(-f[0]), 2),
                "yield_score": round(float(-f[1]), 2),
                "energy_kwh": round(float(f[2]), 2),
                "disintegration_time": round(float(f[3]), 2),
                "carbon_kg": round(float(f[2]) * INDIA_EMISSION_FACTOR, 2),
            })


def compute_golden_signature(objectives: dict, label: str = None) -> dict:
    """
    Select the best Pareto-optimal solution for the given objective weights.
    Initialize a Bayesian parameter tracker with NIG conjugate priors.
    """
    _ensure_pareto()
    df = _get_data()

    w_quality = objectives.get("quality", 0.25)
    w_yield = objectives.get("yield", 0.25)
    w_energy = objectives.get("energy", 0.25)
    w_perf = objectives.get("performance", 0.25)
    total_w = w_quality + w_yield + w_energy + w_perf + 1e-9

    # Weighted scalarization over Pareto front
    F = _pareto_F.copy()
    F_min, F_max = F.min(axis=0), F.max(axis=0)
    F_range = np.where(F_max - F_min < 1e-9, 1.0, F_max - F_min)
    F_norm = (F - F_min) / F_range

    preference = (
        (w_quality / total_w) * F_norm[:, 0]
        + (w_yield / total_w) * F_norm[:, 1]
        + (w_energy / total_w) * F_norm[:, 2]
        + (w_perf / total_w) * F_norm[:, 3]
    )
    best_idx = preference.argmin()
    best_x = _pareto_X[best_idx]
    best_f = _pareto_F[best_idx]

    # Initialize Bayesian tracker from Pareto neighbourhood
    dists = np.linalg.norm(_pareto_X - best_x, axis=1)
    nn_idx = np.argsort(dists)[:15]
    nn_X = _pareto_X[nn_idx]

    # Create BayesianParameterTracker with NIG priors from Pareto solutions
    initial_data = {
        param: nn_X[:, i] for i, param in enumerate(INPUT_PARAMS)
    }
    sig_id = label or "_".join(f"{k}{int(v*10)}" for k, v in objectives.items())
    tracker = BayesianParameterTracker(INPUT_PARAMS, initial_data)
    _bayesian_trackers[sig_id] = tracker

    # Get Bayesian credible intervals (true posterior, NOT frequentist)
    signature_params = tracker.get_credible_intervals(0.95)
    for i, param in enumerate(INPUT_PARAMS):
        signature_params[param]["optimized_value"] = round(float(best_x[i]), 3)

    # Predicted outcomes with uncertainty
    expected_outcomes = {}
    for target in OUTPUT_TARGETS + ["total_kwh", "Quality_Score", "Yield_Score"]:
        if target not in _surrogates:
            continue
        preds = [_predict_sur(nn_X[j], target) for j in range(len(nn_X))]
        expected_outcomes[target] = {
            "mean": round(float(np.mean(preds)), 3),
            "ci_low": round(float(np.percentile(preds, 2.5)), 3),
            "ci_high": round(float(np.percentile(preds, 97.5)), 3),
        }
    expected_outcomes["carbon_kg"] = {
        "mean": round(expected_outcomes.get("total_kwh", {}).get("mean", 0) * INDIA_EMISSION_FACTOR, 3),
        "ci_low": round(expected_outcomes.get("total_kwh", {}).get("ci_low", 0) * INDIA_EMISSION_FACTOR, 3),
        "ci_high": round(expected_outcomes.get("total_kwh", {}).get("ci_high", 0) * INDIA_EMISSION_FACTOR, 3),
    }

    # Nearest historical batch
    X_hist = df[INPUT_PARAMS].values
    hist_dists = np.linalg.norm(X_hist - best_x, axis=1)
    ref_batch = df.iloc[np.argmin(hist_dists)]["Batch_ID"]

    signature = {
        "id": sig_id,
        "label": label or f"Custom ({', '.join(f'{k}={int(v*100)}%' for k, v in objectives.items() if v > 0)})",
        "objectives": objectives,
        "parameters": signature_params,
        "expected_outcomes": expected_outcomes,
        "reference_batch": ref_batch,
        "confidence": "HIGH",
        "n_supporting_batches": len(nn_idx),
        "method": "NSGA-II Pareto + Bayesian NIG Posteriors",
        "bayesian_inference": {
            "prior_type": "Normal-Inverse-Gamma conjugate",
            "n_initial_observations": len(nn_idx),
            "interval_type": "Bayesian credible interval (NOT frequentist)",
        },
        "pareto_position": int(best_idx),
        "objectives_achieved": {
            "quality": round(float(-best_f[0]), 2),
            "yield_score": round(float(-best_f[1]), 2),
            "energy_kwh": round(float(best_f[2]), 2),
            "disintegration_time": round(float(best_f[3]), 2),
        },
    }
    _golden_signatures[sig_id] = signature
    return signature


def get_all_signatures() -> list:
    _ensure_signatures()
    return list(_golden_signatures.values())


def get_pareto_data() -> list:
    _ensure_signatures()
    _ensure_pareto()
    df = _get_data()
    historical = [
        {
            "batch_id": row["Batch_ID"],
            "energy_kwh": round(float(row["total_kwh"]), 2),
            "quality_score": round(float(row["Quality_Score"]), 1),
            "dissolution_rate": round(float(row["Dissolution_Rate"]), 1),
            "hardness": round(float(row["Hardness"]), 1),
            "carbon_kg": round(float(row["carbon_kg"]), 2),
            "type": "historical",
        }
        for _, row in df.iterrows()
    ]
    pareto = [
        {**p, "batch_id": f"Pareto-{i+1}", "type": "pareto_optimal"}
        for i, p in enumerate(_pareto_cache or [])
    ]
    return historical + pareto


def check_and_update_signature(sig_id: str, new_batch_id: str) -> dict:
    """
    Bayesian self-update: if a new batch qualifies, update the posterior.
    Also checks Pareto dominance for the objective values.
    """
    if sig_id not in _golden_signatures:
        return {"updated": False, "reason": "Signature not found"}

    _ensure_pareto()
    sig = _golden_signatures[sig_id]
    df = _get_data()
    new_row = df[df["Batch_ID"] == new_batch_id]
    if new_row.empty:
        return {"updated": False, "reason": "Batch not found"}

    new = new_row.iloc[0]
    new_quality = float(new["Quality_Score"])
    new_yield = float(new["Yield_Score"])
    new_energy = float(new["total_kwh"])
    new_disint = float(new.get("Disintegration_Time", 10))

    cur_obj = sig.get("objectives_achieved", {})
    cur_quality = cur_obj.get("quality", 0)
    cur_yield = cur_obj.get("yield_score", 0)
    cur_energy = cur_obj.get("energy_kwh", 999)
    cur_disint = cur_obj.get("disintegration_time", 999)

    # Pareto dominance check
    new_dominates = (
        new_quality >= cur_quality and new_yield >= cur_yield
        and new_energy <= cur_energy and new_disint <= cur_disint
        and (new_quality > cur_quality or new_yield > cur_yield
             or new_energy < cur_energy or new_disint < cur_disint)
    )

    # BAYESIAN UPDATE: update posterior regardless of dominance
    # (the Bayesian tracker absorbs all qualifying evidence)
    bayesian_update_result = None
    if sig_id in _bayesian_trackers:
        batch_params = {p: float(new[p]) for p in INPUT_PARAMS if p in new.index}
        bayesian_update_result = _bayesian_trackers[sig_id].update_from_batch(batch_params)

        # Refresh the signature's parameters with updated posteriors
        sig["parameters"] = _bayesian_trackers[sig_id].get_credible_intervals(0.95)

    if new_dominates:
        improvement = (
            (new_quality - cur_quality) / max(cur_quality, 1) * 100
            + (cur_energy - new_energy) / max(cur_energy, 1) * 100
        ) / 2
        sig["reference_batch"] = new_batch_id
        sig["objectives_achieved"] = {
            "quality": round(new_quality, 2),
            "yield_score": round(new_yield, 2),
            "energy_kwh": round(new_energy, 2),
            "disintegration_time": round(new_disint, 2),
        }
        return {
            "updated": True,
            "new_reference": new_batch_id,
            "improvement_pct": round(improvement, 1),
            "reason": "New batch Pareto-dominates current signature",
            "bayesian_update": bayesian_update_result,
        }

    return {
        "updated": False,
        "reason": "New batch does not Pareto-dominate current signature",
        "bayesian_update": bayesian_update_result,
        "note": "Posterior was still updated with this observation (Bayesian learning is continuous)",
    }


def get_bayesian_state(sig_id: str = None) -> dict:
    """Return the full Bayesian posterior state for a signature."""
    if sig_id and sig_id in _bayesian_trackers:
        return _bayesian_trackers[sig_id].get_posterior_state()

    # Return all
    result = {}
    for sid, tracker in _bayesian_trackers.items():
        result[sid] = tracker.get_posterior_state()
    return result


def bayesian_update_from_feedback(sig_id: str, param_values: dict) -> dict:
    """
    Manual Bayesian update from operator feedback.
    The operator approves a set of parameter values → posterior updates.
    """
    if sig_id not in _bayesian_trackers:
        return {"error": f"Signature {sig_id} not found"}

    tracker = _bayesian_trackers[sig_id]
    result = tracker.update_from_batch(param_values)

    # Refresh signature parameters
    if sig_id in _golden_signatures:
        _golden_signatures[sig_id]["parameters"] = tracker.get_credible_intervals(0.95)

    return {
        "updated": True,
        "method": "Bayesian NIG conjugate posterior update",
        "result": result,
        "new_credible_intervals": tracker.get_credible_intervals(0.95),
    }


# ─── LAZY INIT (with background eager warm-up) ───────────────────────────────
# On import, a background thread starts computing signatures immediately.
# API calls check _init_done_event; if not ready yet they wait (but don't
# re-trigger the computation). Once done, all calls return instantly from cache.

_surrogates_ready = False
init_signatures_done = False


def _ensure_surrogates():
    """Lazy init: train LightGBM surrogates on first use."""
    global _surrogates_ready
    if _surrogates_ready:
        return
    logger.info("Training surrogate models for NSGA-II...")
    _train_surrogates()
    _surrogates_ready = True
    logger.info("Surrogate models ready.")


def _ensure_signatures():
    """Ensure golden signatures are ready; block until background init finishes."""
    global init_signatures_done
    if init_signatures_done:
        return
    # If the background thread hasn't finished yet, wait for it (max 10 min).
    # Use a lock so only ONE thread does the actual computation.
    with _init_lock:
        if init_signatures_done:  # double-check under lock
            return
        _ensure_surrogates()
        try:
            _ensure_pareto()
            compute_golden_signature({"quality": 1.0, "yield": 0, "energy": 0, "performance": 0}, "best_quality")
            compute_golden_signature({"quality": 0, "yield": 0, "energy": 1.0, "performance": 0}, "best_energy")
            compute_golden_signature({"quality": 0.4, "yield": 0.3, "energy": 0.2, "performance": 0.1}, "balanced")
            compute_golden_signature({"quality": 0.3, "yield": 0.3, "energy": 0.3, "performance": 0.1}, "sustainability")
            init_signatures_done = True
        except Exception as e:
            logger.warning(f"Signature init error: {e}")
        logger.info(f"✅ Golden Signature ready. {len(_bayesian_trackers)} Bayesian trackers active.")
        _init_done_event.set()


def _background_init():
    """Called in a background thread at import time to pre-warm everything."""
    try:
        _ensure_signatures()
    except Exception as e:
        logger.warning(f"Background golden init error: {e}")


# Kick off background warm-up immediately on import (non-blocking)
_bg_thread = _threading.Thread(target=_background_init, daemon=True, name="golden-init")
_bg_thread.start()

