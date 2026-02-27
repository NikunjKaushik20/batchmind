"""
BatchMind CoPilot — Industrial AI Decision Support

Architecture: OpenAI function calling → real ML tool execution
The LLM is a control layer over our actual engines, not a chatbot wrapper.

Tools the LLM can call:
  get_system_status     → live KPIs from all modules
  run_optimization      → NSGA-II with LLM-specified weights
  analyze_counterfactual → DoWhy do-calculus on any batch
  get_carbon_analysis   → carbon budget + forecast
  get_anomaly_report    → DTW-based asset health across fleet
  get_golden_signature  → NSGA-II Pareto signatures
"""
import json
import os
import traceback
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/api/copilot", tags=["copilot"])

# ─── TOOL DEFINITIONS FOR FUNCTION CALLING ───────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_system_status",
            "description": (
                "Get live manufacturing system overview: KPIs, average health, "
                "top/bottom performing batches by quality and energy, carbon summary. "
                "Call this first when operator asks a general question."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_optimization",
            "description": (
                "Run NSGA-II multi-objective causal optimization. Translate the "
                "operator's goal into weights (0-1, all positive, will be normalized). "
                "Use when operator says things like 'prioritize yield', "
                "'save more energy', 'focus on quality', 'maximize dissolution rate'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "quality": {"type": "number", "description": "Weight for tablet quality (0-1)"},
                    "yield_score": {"type": "number", "description": "Weight for yield / dissolution (0-1)"},
                    "energy": {"type": "number", "description": "Weight for energy minimization (0-1)"},
                    "performance": {"type": "number", "description": "Weight for process performance (0-1)"},
                    "reasoning": {"type": "string", "description": "One-sentence reason for these weights"},
                },
                "required": ["quality", "yield_score", "energy", "performance", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_counterfactual",
            "description": (
                "Run causal counterfactual analysis (do-calculus) for a specific batch. "
                "Shows exactly which parameters to intervene on (do(X=x)) to save energy "
                "while maintaining quality. Use when operator mentions a specific batch ID."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "batch_id": {
                        "type": "string",
                        "description": "Batch ID (e.g. T001, T042). Extract from operator message.",
                    }
                },
                "required": ["batch_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_carbon_analysis",
            "description": (
                "Get current carbon emissions status, monthly budget tracking, "
                "and end-of-month forecast. Use when operator asks about sustainability, "
                "carbon budget, emissions, or CO₂."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_anomaly_report",
            "description": (
                "Get DTW-based energy fingerprint anomaly report for all batches. "
                "Identifies equipment health issues (deviations from normal power signature). "
                "Use when operator asks about equipment health, anomalies, or maintenance."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_golden_signature",
            "description": (
                "Get the NSGA-II Pareto-optimal golden signature for a specific goal. "
                "Returns recommended parameters with 95% Bayesian confidence intervals."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "enum": ["best_quality", "best_energy", "balanced", "sustainability"],
                        "description": "The optimization goal preset",
                    }
                },
                "required": ["goal"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are BatchMind CoPilot — the industrial AI decision-support system for pharmaceutical tablet manufacturing at IIT Hyderabad.

You control a real manufacturing intelligence platform with:
- Structural Causal Model (SCM) with nonlinear structural equations (Pearl's framework)
- True do-calculus: P(Y | do(X=x)) via graph surgery, not P(Y|X=x)
- Counterfactuals: Abduction-Action-Prediction (Pearl, 2009) — batch-specific noise U fixed
- DoWhy causal identification + refutation testing (placebo, random common cause, data subset)
- E-value sensitivity analysis for unobserved confounding
- NSGA-II constrained optimization (4 objectives, 4 pharmacopoeial constraints)
- Phase-aware energy fingerprinting (DTW Barycentric Average + Isolation Forest + Ruptures PELT)
- Bayesian golden signatures (Normal-Inverse-Gamma conjugate posteriors, true credible intervals)
- Physics engine: Heckel compression, Ryshkewitch strength, Page drying kinetics (Arrhenius)
- Carbon ledger (India grid: 0.82 kg CO₂e/kWh)
- Data: 60 pharmaceutical tablet batches, minute-level sensor readings

You are NOT a chatbot. You are an industrial co-pilot that:
1. ALWAYS calls a tool before answering system state questions
2. Grounds every answer in real numbers from tool outputs
3. Uses causal language ("intervening on X causes Y decrease in energy")
4. Translates natural language objectives into optimizer weights
5. Identifies the most actionable insight for the operator

Response format rules:
- Lead with the key number or insight
- Then explain causally (1-2 sentences max)
- Then the actionable recommendation
- Keep it under 120 words unless detail is requested
- Use bullet points for lists of parameters
- Be direct. Operators are busy."""


# ─── TOOL EXECUTOR ───────────────────────────────────────────────────────────

def _execute_tool(name: str, args: dict) -> dict:
    try:
        if name == "get_system_status":
            from data_loader import load_production_data, get_all_batch_ids
            from models.fingerprint import get_all_anomaly_scores
            from models.carbon import get_carbon_summary

            prod = load_production_data()
            prod["quality_score"] = (
                (prod["Hardness"].clip(0, 120) / 120) * 30
                + (prod["Dissolution_Rate"].clip(0, 100) / 100) * 30
                + (1 - prod["Friability"].clip(0, 2) / 2) * 15
                + (1 - prod["Disintegration_Time"].clip(0, 20) / 20) * 10
                + (prod["Content_Uniformity"].clip(0, 105) / 105) * 15
            )
            health = get_all_anomaly_scores()
            carbon = get_carbon_summary()

            avg_q = float(prod["quality_score"].mean())
            best = prod.nlargest(3, "quality_score")[["Batch_ID", "quality_score", "Dissolution_Rate"]].to_dict("records")
            worst_q = prod.nsmallest(3, "quality_score")[["Batch_ID", "quality_score"]].to_dict("records")
            worst_health = sorted(health, key=lambda x: x["overall_health"])[:3]

            return {
                "n_batches": len(prod),
                "avg_quality_score": round(avg_q, 1),
                "top_quality_batches": best,
                "lowest_quality_batches": worst_q,
                "lowest_health_batches": worst_health,
                "carbon_summary": carbon,
            }

        elif name == "run_optimization":
            from models.causal import causal_optimize
            objectives = {
                "quality": float(args["quality"]),
                "yield": float(args["yield_score"]),
                "energy": float(args["energy"]),
                "performance": float(args["performance"]),
            }
            result = causal_optimize(objectives)
            # Trim for LLM context
            return {
                "objectives_requested": objectives,
                "reasoning": args.get("reasoning", ""),
                "method": result.get("method"),
                "confidence": result.get("confidence"),
                "n_pareto_solutions": result.get("n_pareto_solutions"),
                "best_objectives": result.get("best_objectives"),
                "predicted_outcomes": result.get("predicted_outcomes"),
                "top_recommendations": {
                    k: v for i, (k, v) in enumerate(result.get("recommended_params", {}).items()) if i < 5
                },
            }

        elif name == "analyze_counterfactual":
            from models.causal import counterfactual
            cf = counterfactual(args["batch_id"])
            return {
                "batch_id": cf.get("batch_id"),
                "actual_energy_kwh": cf.get("actual_energy_kwh"),
                "counterfactual_energy_kwh": cf.get("counterfactual_energy_kwh"),
                "energy_saved_kwh": cf.get("energy_saved_kwh"),
                "pct_energy_saved": cf.get("pct_energy_saved"),
                "carbon_saved_kg": cf.get("carbon_saved_kg"),
                "quality_maintained": cf.get("quality_maintained"),
                "method": cf.get("method"),
                "top_interventions": {
                    k: v for i, (k, v) in enumerate(cf.get("parameter_changes", {}).items()) if i < 4
                },
            }

        elif name == "get_carbon_analysis":
            from models.carbon import get_carbon_summary, get_carbon_forecast
            summary = get_carbon_summary()
            forecast = get_carbon_forecast()
            return {"summary": summary, "forecast": forecast}

        elif name == "get_anomaly_report":
            from models.fingerprint import get_all_anomaly_scores
            scores = get_all_anomaly_scores()
            scores_sorted = sorted(scores, key=lambda x: x["overall_health"])
            return {
                "fleet_avg_health": round(
                    sum(s["overall_health"] for s in scores) / len(scores), 1
                ),
                "critical_batches": [
                    s for s in scores_sorted[:5] if s["overall_health"] < 70
                ],
                "warning_batches": [
                    s for s in scores_sorted if 70 <= s["overall_health"] < 85
                ][:5],
                "method": "DTW phase fingerprint deviation",
                "total_batches_analyzed": len(scores),
            }

        elif name == "get_golden_signature":
            from models.golden_signature import get_all_signatures
            sigs = get_all_signatures()
            goal = args.get("goal", "balanced")
            sig = next((s for s in sigs if s["id"] == goal), sigs[0] if sigs else {})
            return {
                "id": sig.get("id"),
                "label": sig.get("label"),
                "confidence": sig.get("confidence"),
                "objectives_achieved": sig.get("objectives_achieved"),
                "top_parameters": {
                    k: {"value": v["value"], "ci_low": v["ci_low"], "ci_high": v["ci_high"]}
                    for i, (k, v) in enumerate(sig.get("parameters", {}).items()) if i < 5
                },
                "method": sig.get("method"),
            }

        return {"error": f"Unknown tool: {name}"}

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()[-300:]}


# ─── REQUEST / RESPONSE MODELS ────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class CopilotRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    context: dict = {}


class ToolResult(BaseModel):
    name: str
    data: dict


class CopilotResponse(BaseModel):
    reply: str
    tool_results: list[dict] = []
    error: Optional[str] = None


# ─── MAIN CHAT HANDLER ───────────────────────────────────────────────────────

@router.post("/chat")
def chat(req: CopilotRequest) -> CopilotResponse:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key in ("your_openai_api_key_here", ""):
        # Intelligent fallback without OpenAI
        return _fallback_response(req.message, req.context)

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add context if available
    if req.context:
        ctx_str = f"Current operator context: {json.dumps(req.context)}"
        messages.append({"role": "system", "content": ctx_str})

    # Add conversation history (last 6 turns)
    for msg in req.history[-6:]:
        messages.append({"role": msg.role, "content": msg.content})

    messages.append({"role": "user", "content": req.message})

    tool_results = []
    max_tool_rounds = 3  # prevent infinite loop

    for _ in range(max_tool_rounds):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=600,
        )

        choice = response.choices[0]

        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            # Append assistant's tool call message
            messages.append(choice.message)

            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}

                result = _execute_tool(tc.function.name, args)
                tool_results.append({"name": tc.function.name, "args": args, "data": result})

                # Feed result back to LLM
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })
        else:
            # Final text response
            return CopilotResponse(
                reply=choice.message.content,
                tool_results=tool_results,
            )

    return CopilotResponse(
        reply="I've gathered the data. Here's the summary from the tool results above.",
        tool_results=tool_results,
    )


def _fallback_response(message: str, context: dict) -> CopilotResponse:
    """Intelligent rule-based fallback when no OpenAI key."""
    from models.fingerprint import get_all_anomaly_scores
    from models.carbon import get_carbon_summary
    from data_loader import load_production_data

    msg = message.lower()
    tool_results = []

    if any(w in msg for w in ["energy", "power", "kwh", "consumption"]):
        scores = get_all_anomaly_scores()
        worst = sorted(scores, key=lambda x: x["overall_health"])[:3]
        tool_results.append({"name": "get_anomaly_report", "data": {"critical_batches": worst}})
        reply = (
            f"Energy anomaly scan complete (DTW fingerprinting). "
            f"Lowest asset health: {worst[0]['batch_id']} at {worst[0]['overall_health']:.0f}%. "
            f"DTW deviation from normal power signature indicates equipment degradation. "
            f"Recommend causal counterfactual analysis on these batches."
        )
    elif any(w in msg for w in ["carbon", "co2", "emission", "green", "sustainability"]):
        carbon = get_carbon_summary()
        from models.carbon import get_carbon_forecast
        forecast = get_carbon_forecast()
        tool_results.append({"name": "get_carbon_analysis", "data": {"summary": carbon, "forecast": forecast}})
        reply = (
            f"Carbon status: {carbon.get('total_carbon_kg', 0):.0f} kg CO₂e total across "
            f"{carbon.get('n_batches', 60)} batches. "
            f"Budget utilisation: {carbon.get('budget_used_pct', 0):.0f}%. "
            f"Month-end forecast: {forecast.get('projected_month_end_kg', 0):.0f} kg CO₂e. "
            f"Potential savings via NSGA-II optimization: {carbon.get('potential_savings_kg', 0):.0f} kg CO₂e."
        )
    elif any(w in msg for w in ["quality", "hardness", "dissolution", "friability"]):
        prod = load_production_data()
        prod["qs"] = (prod["Hardness"].clip(0,120)/120)*30 + (prod["Dissolution_Rate"].clip(0,100)/100)*30
        top = prod.nlargest(3, "qs")[["Batch_ID","Hardness","Dissolution_Rate"]].to_dict("records")
        reply = (
            f"Top quality batches: {', '.join(b['Batch_ID'] for b in top)}. "
            f"Best dissolution rate: {max(b['Dissolution_Rate'] for b in top):.1f}%. "
            f"Use Golden Signature → Best Quality to get NSGA-II optimal parameters with 95% CI."
        )
    else:
        reply = (
            "I'm BatchMind CoPilot. I can help you:\n"
            "• **Optimize** — 'Prioritize yield this week'\n"
            "• **Investigate** — 'Why did T042 use so much energy?'\n"
            "• **Monitor** — 'Which batches have equipment issues?'\n"
            "• **Plan** — 'What's our carbon forecast?'\n\n"
            "Add your OpenAI API key to `.env` for full AI capabilities."
        )

    return CopilotResponse(reply=reply, tool_results=tool_results)


@router.get("/health")
def copilot_health():
    api_key = os.getenv("OPENAI_API_KEY", "")
    has_key = bool(api_key) and api_key not in ("your_openai_api_key_here", "")
    return {"copilot_ready": has_key, "mode": "gpt-4o-mini" if has_key else "rule-based fallback"}
