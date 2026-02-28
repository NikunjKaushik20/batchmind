"""
Adaptive Constraint Engine — Dynamic Manufacturing Constraint Tightening

Implements performance-based constraint adaptation for NSGA-II:
  1. Monitors recent batch performance against current constraints
  2. Tightens constraints when batches consistently exceed targets
  3. Relaxes constraints when feasibility rate drops below threshold
  4. Maintains constraint history for audit trail

Mathematics:
  For each constraint C with bound b:
    Achievement ratio: r = achieved_value / bound
    If percentile_10(r, recent_batches) > tighten_threshold:
      b_new = b + tighten_rate × (percentile_10 - b)
    If feasibility_rate < relax_threshold:
      b_new = b - relax_rate × b

References:
  Deb, K. & Gupta, H. (2006). Introducing robustness in multi-objective optimization.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ConstraintSpec:
    """A single manufacturing constraint with adaptive bounds."""
    name: str
    bound_type: str   # 'min' or 'max'
    initial_bound: float
    current_bound: float
    unit: str
    tighten_rate: float = 0.3       # fraction of gap to tighten
    relax_rate: float = 0.1         # fraction to relax
    tighten_threshold: float = 1.3  # 10th percentile must exceed bound by 30%
    relax_threshold: float = 0.5    # feasibility rate below 50% triggers relax
    min_bound: float = None         # can't tighten beyond this
    max_bound: float = None         # can't relax beyond this
    history: list = field(default_factory=list)

    def record_update(self, old_bound, new_bound, reason, n_batches):
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "old_bound": round(old_bound, 4),
            "new_bound": round(new_bound, 4),
            "reason": reason,
            "n_batches_considered": n_batches,
        })


class AdaptiveConstraintManager:
    """
    Dynamically adjusts manufacturing constraints based on recent batch performance.

    Architecture:
      1. Each constraint has a bound (min or max) and adaptation parameters
      2. On update(), recent batches are scored against constraints
      3. If performance consistently exceeds a constraint, it tightens
      4. If feasibility drops too low, it relaxes

    This is NOT static thresholds — constraints evolve with plant capability.
    The adaptation rate is configurable per-constraint for fine-grained control.
    """

    def __init__(self, window_size: int = 20):
        """
        Args:
            window_size: Number of recent batches to consider for adaptation.
        """
        self.window_size = window_size
        self.constraints: Dict[str, ConstraintSpec] = {}
        self.adaptation_count = 0
        self._initialize_defaults()

    def _initialize_defaults(self):
        """Initialize with pharmacopoeial baseline constraints."""
        defaults = [
            ConstraintSpec(
                name="Hardness", bound_type="min",
                initial_bound=40.0, current_bound=40.0, unit="N",
                tighten_rate=0.3, min_bound=40.0, max_bound=90.0,
            ),
            ConstraintSpec(
                name="Friability", bound_type="max",
                initial_bound=1.0, current_bound=1.0, unit="%",
                tighten_rate=0.25, min_bound=0.3, max_bound=1.0,
            ),
            ConstraintSpec(
                name="Dissolution_Rate", bound_type="min",
                initial_bound=80.0, current_bound=80.0, unit="%",
                tighten_rate=0.3, min_bound=80.0, max_bound=98.0,
            ),
            ConstraintSpec(
                name="Disintegration_Time", bound_type="max",
                initial_bound=15.0, current_bound=15.0, unit="min",
                tighten_rate=0.25, min_bound=5.0, max_bound=15.0,
            ),
        ]
        for c in defaults:
            self.constraints[c.name] = c

    def update(self, recent_batches: pd.DataFrame) -> dict:
        """
        Adapt constraints based on recent batch performance.

        For 'min' constraints (e.g., Hardness ≥ 40):
          - If 10th percentile of recent values > bound × tighten_threshold:
            tighten (plant is consistently exceeding)
          - If < relax_threshold fraction of batches meet constraint:
            relax (constraint is too aggressive)

        For 'max' constraints (e.g., Friability ≤ 1.0):
          - If 90th percentile of recent values < bound × (2 - tighten_threshold):
            tighten
          - If feasibility drops: relax

        Returns dict of all constraint changes.
        """
        if len(recent_batches) < 5:
            return {"adapted": False, "reason": "insufficient data (need ≥5 batches)"}

        # Use most recent window_size batches
        df = recent_batches.tail(self.window_size)
        n = len(df)
        changes = {}

        for name, spec in self.constraints.items():
            if name not in df.columns:
                continue

            values = df[name].dropna().values
            if len(values) < 3:
                continue

            old_bound = spec.current_bound

            if spec.bound_type == "min":
                # Constraint: value ≥ bound
                feasibility = float(np.mean(values >= spec.current_bound))
                p10 = float(np.percentile(values, 10))
                ratio = p10 / spec.current_bound if spec.current_bound > 0 else 1.0

                if ratio > spec.tighten_threshold and feasibility > 0.9:
                    # Plant consistently exceeds — tighten
                    gap = p10 - spec.current_bound
                    new_bound = spec.current_bound + spec.tighten_rate * gap
                    if spec.max_bound is not None:
                        new_bound = min(new_bound, spec.max_bound)
                    spec.current_bound = round(new_bound, 2)
                    spec.record_update(old_bound, spec.current_bound,
                                       "tightened: plant consistently exceeds", n)
                    changes[name] = {
                        "action": "tightened",
                        "old": old_bound, "new": spec.current_bound,
                        "feasibility": round(feasibility, 3),
                        "p10": round(p10, 2),
                    }
                elif feasibility < spec.relax_threshold:
                    # Too many failures — relax toward initial
                    relax_step = spec.relax_rate * (spec.current_bound - spec.initial_bound)
                    new_bound = spec.current_bound - max(relax_step, 1.0)
                    new_bound = max(new_bound, spec.initial_bound)
                    if spec.min_bound is not None:
                        new_bound = max(new_bound, spec.min_bound)
                    spec.current_bound = round(new_bound, 2)
                    spec.record_update(old_bound, spec.current_bound,
                                       "relaxed: feasibility too low", n)
                    changes[name] = {
                        "action": "relaxed",
                        "old": old_bound, "new": spec.current_bound,
                        "feasibility": round(feasibility, 3),
                    }

            elif spec.bound_type == "max":
                # Constraint: value ≤ bound
                feasibility = float(np.mean(values <= spec.current_bound))
                p90 = float(np.percentile(values, 90))
                ratio = p90 / spec.current_bound if spec.current_bound > 0 else 1.0

                if ratio < (2 - spec.tighten_threshold) and feasibility > 0.9:
                    # Plant consistently well below — tighten
                    gap = spec.current_bound - p90
                    new_bound = spec.current_bound - spec.tighten_rate * gap
                    if spec.min_bound is not None:
                        new_bound = max(new_bound, spec.min_bound)
                    spec.current_bound = round(new_bound, 2)
                    spec.record_update(old_bound, spec.current_bound,
                                       "tightened: plant consistently below limit", n)
                    changes[name] = {
                        "action": "tightened",
                        "old": old_bound, "new": spec.current_bound,
                        "feasibility": round(feasibility, 3),
                        "p90": round(p90, 2),
                    }
                elif feasibility < spec.relax_threshold:
                    relax_step = spec.relax_rate * (spec.initial_bound - spec.current_bound)
                    new_bound = spec.current_bound + max(relax_step, 0.1)
                    new_bound = min(new_bound, spec.initial_bound)
                    if spec.max_bound is not None:
                        new_bound = min(new_bound, spec.max_bound)
                    spec.current_bound = round(new_bound, 2)
                    spec.record_update(old_bound, spec.current_bound,
                                       "relaxed: feasibility too low", n)
                    changes[name] = {
                        "action": "relaxed",
                        "old": old_bound, "new": spec.current_bound,
                        "feasibility": round(feasibility, 3),
                    }

        if changes:
            self.adaptation_count += 1
            logger.info(f"Constraints adapted (round {self.adaptation_count}): {changes}")

        return {
            "adapted": bool(changes),
            "changes": changes,
            "current_constraints": self.get_current_bounds(),
            "adaptation_round": self.adaptation_count,
            "n_batches_considered": n,
        }

    def get_current_bounds(self) -> dict:
        """Return current constraint bounds for NSGA-II."""
        return {
            name: {
                "bound": spec.current_bound,
                "type": spec.bound_type,
                "initial": spec.initial_bound,
                "deviation_from_initial": round(
                    (spec.current_bound - spec.initial_bound) / spec.initial_bound * 100, 1
                ) if spec.initial_bound != 0 else 0,
                "unit": spec.unit,
            }
            for name, spec in self.constraints.items()
        }

    def get_nsga2_constraints(self) -> dict:
        """Return bounds in format ready for NSGA-II problem."""
        return {name: spec.current_bound for name, spec in self.constraints.items()}

    def get_history(self) -> dict:
        """Return full adaptation history for audit."""
        return {
            name: spec.history for name, spec in self.constraints.items()
        }

    def get_state(self) -> dict:
        """Full state for API/visualization."""
        return {
            "constraints": {
                name: {
                    "current_bound": spec.current_bound,
                    "initial_bound": spec.initial_bound,
                    "bound_type": spec.bound_type,
                    "unit": spec.unit,
                    "tighten_rate": spec.tighten_rate,
                    "n_adaptations": len(spec.history),
                    "history": spec.history[-5:],  # last 5 changes
                }
                for name, spec in self.constraints.items()
            },
            "total_adaptations": self.adaptation_count,
            "window_size": self.window_size,
        }


# Module-level singleton
_constraint_manager = None


def get_constraint_manager() -> AdaptiveConstraintManager:
    global _constraint_manager
    if _constraint_manager is None:
        _constraint_manager = AdaptiveConstraintManager(window_size=20)
    return _constraint_manager
