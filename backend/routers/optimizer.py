from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from models.causal import (
    causal_optimize, counterfactual, get_causal_graph,
    get_shap_importance, get_causal_effects, get_scm_info,
    get_refutation_results, get_sensitivity_analysis,
)
import os
from openai import OpenAI

router = APIRouter(prefix="/api/optimize", tags=["optimizer"])

_openai_client = None


def get_openai():
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key and api_key not in ("your_openai_api_key_here", ""):
            _openai_client = OpenAI(api_key=api_key)
    return _openai_client


class OptimizeRequest(BaseModel):
    quality: float = 0.25
    yield_score: float = 0.25
    energy: float = 0.25
    performance: float = 0.25


@router.post("")
def optimize(req: OptimizeRequest):
    objectives = {
        "quality": req.quality, "yield": req.yield_score,
        "energy": req.energy, "performance": req.performance,
    }
    try:
        result = causal_optimize(objectives)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/counterfactual/{batch_id}")
def get_counterfactual(batch_id: str):
    try:
        result = counterfactual(batch_id)
        if not result:
            raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/explain/{batch_id}")
def explain_counterfactual(batch_id: str):
    try:
        cf = counterfactual(batch_id)
        if not cf:
            raise HTTPException(status_code=404, detail="Batch not found")

        client = get_openai()
        changes_text = "\n".join(
            f"- {param}: {v['actual']} → {v['counterfactual']} ({v['direction']}, "
            f"causal effect on energy: {v.get('causal_effect_on_energy', 'unknown')})"
            for param, v in cf.get("parameter_changes", {}).items()
        )

        # Include SCM counterfactual detail
        scm_detail = ""
        scm_cf = cf.get("scm_counterfactual", {})
        if scm_cf.get("effects"):
            scm_detail = "\nSCM counterfactual effects (Pearl's 3-step):\n"
            for var, eff in list(scm_cf["effects"].items())[:5]:
                scm_detail += f"  {var}: {eff.get('observed', '?')} → {eff.get('counterfactual', '?')} (Δ={eff.get('effect', '?')})\n"

        if not client:
            return {
                "explanation": (
                    f"Batch {cf['batch_id']} consumed {cf['actual_energy_kwh']} kWh. "
                    f"Our causal analysis (Abduction-Action-Prediction via SCM + NSGA-II) shows that "
                    f"intervening on {', '.join(list(cf.get('parameter_changes', {}).keys())[:3])} — "
                    f"the same quality could have been achieved at {cf['counterfactual_energy_kwh']} kWh, "
                    f"saving {cf['energy_saved_kwh']} kWh ({cf['pct_energy_saved']}%) "
                    f"and {cf['carbon_saved_kg']} kg CO₂e. "
                    f"Quality maintained: {cf['actual_quality']:.1f} vs {cf['counterfactual_quality']:.1f}."
                ),
                "source": "template"
            }

        prompt = f"""You are a manufacturing AI expert. A pharma tablet batch has been analysed using a real Structural Causal Model (Pearl's framework).

Batch {cf['batch_id']} analysis:
- Actual energy: {cf['actual_energy_kwh']} kWh
- Counterfactual energy (do-calculus via SCM): {cf['counterfactual_energy_kwh']} kWh
- Energy saving: {cf['energy_saved_kwh']} kWh ({cf['pct_energy_saved']}%)
- Carbon saving: {cf['carbon_saved_kg']} kg CO₂e
- Quality maintained: {cf['actual_quality']:.1f} → {cf['counterfactual_quality']:.1f}
- Method: {cf.get('method', 'SCM + NSGA-II')}

Causal interventions (do-calculus / graph surgery):
{changes_text}
{scm_detail}

Write a clear, operator-friendly 3-sentence explanation:
1. What this batch actually did
2. The causal intervention — use language like "intervening on X causes Y"
3. The measurable impact
Max 80 words."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        return {
            "explanation": response.choices[0].message.content,
            "source": "gpt-4o-mini"
        }
    except HTTPException:
        raise
    except Exception as e:
        return {"explanation": f"Analysis unavailable: {str(e)}", "source": "error"}


@router.get("/graph")
def get_graph():
    return get_causal_graph()


@router.get("/importance")
def get_importance():
    return get_shap_importance()


@router.get("/causal-effects")
def causal_effects():
    """Return DoWhy causal effects + refutation results for all treatment-outcome pairs."""
    return get_causal_effects()


# ─── NEW ENDPOINTS ───────────────────────────────────────────────────────────

@router.get("/scm")
def scm_info():
    """Return full SCM: structural equations, R², DAG validation, physics."""
    return get_scm_info()


@router.get("/refutation")
def refutation_results():
    """Return DoWhy refutation test results (placebo, random cause, data subset)."""
    return get_refutation_results()


@router.get("/sensitivity/{treatment}/{outcome}")
def sensitivity(treatment: str, outcome: str):
    """E-value sensitivity analysis for unobserved confounding."""
    return get_sensitivity_analysis(treatment, outcome)
