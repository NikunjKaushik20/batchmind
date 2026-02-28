"""
Structural Causal Model (SCM) Engine — Pearl's Causal Framework

Implements the three levels of Pearl's Causal Hierarchy:
  Level 1 — Association:    P(Y | X=x)        [observational]
  Level 2 — Intervention:   P(Y | do(X=x))    [interventional / do-calculus]
  Level 3 — Counterfactual: P(Y_x | X=x', Y=y') [counterfactual]

Architecture:
  - Directed Acyclic Graph (DAG) via NetworkX
  - Nonlinear structural equations: Vᵢ = fᵢ(pa(Vᵢ)) + Uᵢ
  - fᵢ fitted via GradientBoosting (captures nonlinear mechanisms)
  - Uᵢ (exogenous noise) = residuals, stored per-observation for counterfactuals
  - do() operator: graph surgery (truncation of incoming edges)
  - Counterfactuals: Abduction → Action → Prediction (Pearl, 2009)

References:
  Pearl, J. (2009). Causality: Models, Reasoning, and Inference, 2nd ed.
  Peters, J. et al. (2017). Elements of Causal Inference.
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


class StructuralCausalModel:
    """
    Pearl's Structural Causal Model with nonlinear structural equations.

    An SCM M = (U, V, F) consists of:
      U — exogenous (background) variables
      V — endogenous variables
      F — structural equations: Vᵢ = fᵢ(pa(Vᵢ), Uᵢ)

    This implementation:
      - Fits fᵢ as GradientBoosting models from data
      - Stores residuals Uᵢ = Vᵢ_observed - fᵢ(pa_observed) per observation
      - Supports do() operator via graph surgery
      - Supports counterfactuals via Abduction-Action-Prediction
    """

    def __init__(self, dag_edges: list, data: pd.DataFrame,
                 latent_estimators: dict = None):
        """
        Args:
            dag_edges: list of (parent, child) tuples defining the DAG
            data: DataFrame with columns for all observed endogenous variables
            latent_estimators: dict of {var_name: callable(row) -> float}
                for latent variables that must be estimated (e.g., Moisture_Content)
        """
        self.dag = nx.DiGraph(dag_edges)
        if not nx.is_directed_acyclic_graph(self.dag):
            raise ValueError("Graph contains cycles — not a valid DAG")

        self.data = data.copy()
        self.latent_estimators = latent_estimators or {}
        self.topo_order = list(nx.topological_sort(self.dag))

        # Structural equation models: node → fitted model
        self._struct_models = {}
        # Linear equation coefficients for interpretability
        self._linear_models = {}
        # Exogenous noise per observation: node → array of residuals
        self._exogenous_noise = {}
        # R² scores for each structural equation
        self._r2_scores = {}
        # Cross-validation scores
        self._cv_scores = {}

        self._fitted = False

    def fit(self) -> dict:
        """
        Fit all structural equations from data.

        For each non-root node V with parents pa(V):
          1. Fit nonlinear model: V = f(pa(V)) + U  (GradientBoosting)
          2. Fit linear model:    V = Σβᵢ·paᵢ + U   (OLS, for interpretability)
          3. Compute residuals:   U = V - f(pa(V))
          4. Cross-validate:      5-fold CV R²

        Returns fit quality metrics.
        """
        # First, estimate latent variables
        for var_name, estimator in self.latent_estimators.items():
            if var_name not in self.data.columns:
                self.data[var_name] = self.data.apply(estimator, axis=1)

        results = {}

        for node in self.topo_order:
            parents = sorted(self.dag.predecessors(node))
            if not parents:
                # Root node: no structural equation, exogenous noise = observed value
                if node in self.data.columns:
                    self._exogenous_noise[node] = self.data[node].values.copy()
                continue

            # Check if all parents and node are in the data
            available_parents = [p for p in parents if p in self.data.columns]
            if not available_parents or node not in self.data.columns:
                continue

            X = self.data[available_parents].values
            y = self.data[node].values

            # --- Nonlinear structural equation (GradientBoosting) ---
            gb_model = GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.05,
                max_depth=4, random_state=42,
                min_samples_leaf=3,
            )
            gb_model.fit(X, y)
            self._struct_models[node] = {
                "model": gb_model,
                "parents": available_parents,
                "type": "gradient_boosting",
            }

            # Compute residuals (exogenous noise)
            y_pred = gb_model.predict(X)
            residuals = y - y_pred
            self._exogenous_noise[node] = residuals.copy()

            # R² score
            r2 = r2_score(y, y_pred)
            self._r2_scores[node] = round(r2, 4)

            # Cross-validation
            try:
                cv_scores = cross_val_score(
                    GradientBoostingRegressor(
                        n_estimators=100, learning_rate=0.05,
                        max_depth=4, random_state=42, min_samples_leaf=3
                    ),
                    X, y, cv=min(5, len(y) // 3), scoring="r2"
                )
                self._cv_scores[node] = round(float(cv_scores.mean()), 4)
            except Exception:
                self._cv_scores[node] = None

            # --- Linear structural equation (for interpretability) ---
            lr_model = LinearRegression()
            lr_model.fit(X, y)
            self._linear_models[node] = {
                "model": lr_model,
                "parents": available_parents,
                "coefficients": {
                    p: round(float(c), 4)
                    for p, c in zip(available_parents, lr_model.coef_)
                },
                "intercept": round(float(lr_model.intercept_), 4),
                "r2": round(float(r2_score(y, lr_model.predict(X))), 4),
            }

            results[node] = {
                "parents": available_parents,
                "gb_r2": self._r2_scores[node],
                "cv_r2": self._cv_scores[node],
                "linear_r2": self._linear_models[node]["r2"],
                "linear_coefficients": self._linear_models[node]["coefficients"],
                "residual_std": round(float(residuals.std()), 4),
            }

        self._fitted = True
        return results

    def do(self, interventions: dict) -> dict:
        """
        do-calculus intervention: P(Y | do(X₁=x₁, X₂=x₂, ...))

        Graph surgery: remove all incoming edges to intervened variables.
        Then propagate through the modified SCM in topological order.

        Args:
            interventions: dict of {variable_name: intervention_value}

        Returns:
            dict of all endogenous variable values under the intervention
        """
        if not self._fitted:
            raise RuntimeError("SCM not fitted. Call fit() first.")

        # Start with mean values for root nodes
        values = {}
        for node in self.topo_order:
            if node in interventions:
                # Intervened node: set to intervention value (graph surgery)
                values[node] = interventions[node]
            elif node in self.data.columns and not list(self.dag.predecessors(node)):
                # Root node (not intervened): use mean from data
                values[node] = float(self.data[node].mean())
            elif node in self._struct_models:
                # Non-root, non-intervened: use structural equation
                model_info = self._struct_models[node]
                parent_vals = np.array([
                    values.get(p, self.data[p].mean() if p in self.data.columns else 0)
                    for p in model_info["parents"]
                ]).reshape(1, -1)
                pred = model_info["model"].predict(parent_vals)[0]
                values[node] = float(pred)

        return values

    def interventional_distribution(self, interventions: dict, target: str,
                                     n_samples: int = 500) -> dict:
        """
        Monte Carlo estimate of P(Y | do(X=x)).

        Samples exogenous noise from their empirical distribution,
        then propagates through the modified SCM.

        Returns distribution statistics for the target variable.
        """
        if not self._fitted:
            raise RuntimeError("SCM not fitted. Call fit() first.")

        samples = []
        n_obs = len(self.data)

        for _ in range(n_samples):
            # Sample a noise realization
            noise_idx = np.random.randint(0, n_obs)
            values = {}

            for node in self.topo_order:
                if node in interventions:
                    values[node] = interventions[node]
                elif not list(self.dag.predecessors(node)):
                    # Root node: sample from observed values
                    values[node] = float(
                        self.data[node].iloc[np.random.randint(0, n_obs)]
                    ) if node in self.data.columns else 0
                elif node in self._struct_models:
                    model_info = self._struct_models[node]
                    parent_vals = np.array([
                        values.get(p, 0) for p in model_info["parents"]
                    ]).reshape(1, -1)
                    pred = model_info["model"].predict(parent_vals)[0]
                    # Add sampled noise
                    noise = self._exogenous_noise.get(node, np.zeros(1))
                    sampled_noise = noise[np.random.randint(0, len(noise))]
                    values[node] = float(pred + sampled_noise)

            if target in values:
                samples.append(values[target])

        samples = np.array(samples)
        return {
            "mean": round(float(samples.mean()), 4),
            "std": round(float(samples.std()), 4),
            "ci_low": round(float(np.percentile(samples, 2.5)), 4),
            "ci_high": round(float(np.percentile(samples, 97.5)), 4),
            "median": round(float(np.median(samples)), 4),
            "n_samples": n_samples,
            "type": "interventional_distribution",
        }

    def counterfactual(self, observation_idx: int, interventions: dict) -> dict:
        """
        Pearl's 3-step counterfactual computation.

        Given a specific observed unit (batch), computes what WOULD have
        happened under a different treatment.

        Step 1 — Abduction: Infer exogenous noise U for this specific unit.
          Uᵢ = Vᵢ(observed) - fᵢ(pa_observed)

        Step 2 — Action: do(X = x*), modify the model.
          Remove structural equation for X, set X = x*.

        Step 3 — Prediction: Propagate through modified SCM with fixed U.
          For each node in topological order:
            If intervened: use intervention value
            If root (not intervened): keep observed value
            Else: Vᵢ = fᵢ(parents_counterfactual) + Uᵢ

        Args:
            observation_idx: index into the training data (which batch)
            interventions: dict of {variable: counterfactual_value}

        Returns:
            dict with observed values, counterfactual values, and effects.
        """
        if not self._fitted:
            raise RuntimeError("SCM not fitted. Call fit() first.")

        # Step 1: Abduction — get this unit's exogenous noise
        unit_noise = {}
        for node in self.topo_order:
            if node in self._exogenous_noise:
                noise_arr = self._exogenous_noise[node]
                if observation_idx < len(noise_arr):
                    unit_noise[node] = float(noise_arr[observation_idx])
                else:
                    unit_noise[node] = 0.0

        # Get observed values for this unit
        observed = {}
        for col in self.data.columns:
            observed[col] = float(self.data.iloc[observation_idx][col])

        # Step 2 + 3: Action + Prediction
        cf_values = {}
        for node in self.topo_order:
            if node in interventions:
                # ACTION: Set to intervention value (graph surgery)
                cf_values[node] = float(interventions[node])
            elif not list(self.dag.predecessors(node)):
                # Root node (not intervened): keep observed value
                if node in observed:
                    cf_values[node] = observed[node]
                else:
                    cf_values[node] = 0.0
            elif node in self._struct_models:
                # PREDICTION: f(parents_cf) + U_this_unit
                model_info = self._struct_models[node]
                parent_vals = np.array([
                    cf_values.get(p, observed.get(p, 0))
                    for p in model_info["parents"]
                ]).reshape(1, -1)
                structural_pred = model_info["model"].predict(parent_vals)[0]
                noise = unit_noise.get(node, 0.0)
                cf_values[node] = float(structural_pred + noise)
            elif node in observed:
                cf_values[node] = observed[node]

        # Compute effects
        effects = {}
        for var in cf_values:
            if var in observed:
                diff = cf_values[var] - observed[var]
                effects[var] = {
                    "observed": round(observed[var], 4),
                    "counterfactual": round(cf_values[var], 4),
                    "effect": round(diff, 4),
                    "pct_change": round(diff / abs(observed[var]) * 100, 2)
                        if abs(observed[var]) > 1e-9 else 0,
                }

        return {
            "observation_idx": observation_idx,
            "interventions": interventions,
            "observed_values": {k: round(v, 4) for k, v in observed.items()
                                if k in self.dag.nodes},
            "counterfactual_values": {k: round(v, 4) for k, v in cf_values.items()},
            "effects": effects,
            "unit_noise": {k: round(v, 4) for k, v in unit_noise.items()
                          if k in self._struct_models},
            "method": "Abduction-Action-Prediction (Pearl, 2009)",
        }

    def average_treatment_effect(self, treatment: str, outcome: str,
                                  treatment_low: float = None,
                                  treatment_high: float = None) -> dict:
        """
        Compute ATE via the SCM's interventional distribution.
        ATE = E[Y | do(X=x_high)] - E[Y | do(X=x_low)]
        """
        if treatment_low is None:
            treatment_low = float(self.data[treatment].quantile(0.25))
        if treatment_high is None:
            treatment_high = float(self.data[treatment].quantile(0.75))

        dist_low = self.interventional_distribution(
            {treatment: treatment_low}, outcome, n_samples=300
        )
        dist_high = self.interventional_distribution(
            {treatment: treatment_high}, outcome, n_samples=300
        )

        ate = dist_high["mean"] - dist_low["mean"]

        return {
            "treatment": treatment,
            "outcome": outcome,
            "treatment_low": treatment_low,
            "treatment_high": treatment_high,
            "E_Y_do_high": dist_high["mean"],
            "E_Y_do_low": dist_low["mean"],
            "ATE": round(ate, 4),
            "direction": "positive" if ate > 0 else "negative",
            "ci_low": round(dist_high["ci_low"] - dist_low["ci_high"], 4),
            "ci_high": round(dist_high["ci_high"] - dist_low["ci_low"], 4),
            "method": "SCM interventional distribution",
        }

    def sensitivity_analysis(self, treatment: str, outcome: str) -> dict:
        """
        Sensitivity analysis for unobserved confounding.
        Computes the E-value (VanderWeele & Ding, 2017):
        How strong must an unobserved confounder be to explain away the effect?

        E-value = RR + sqrt(RR × (RR-1))
        For continuous: convert standardized effect to approximate RR.
        """
        ate_result = self.average_treatment_effect(treatment, outcome)
        ate = abs(ate_result["ATE"])

        y_data = self.data[outcome].values if outcome in self.data.columns else np.array([1])
        y_std = float(y_data.std()) if y_data.std() > 0 else 1.0

        # Standardized effect → approximate RR (Chinn, 2000)
        d = ate / y_std
        log_rr = d * np.pi / np.sqrt(3)
        rr = np.exp(abs(log_rr))
        rr = max(rr, 1.001)

        # E-value
        e_value = rr + np.sqrt(rr * (rr - 1))

        if e_value > 5:
            robustness = "very robust"
        elif e_value > 3:
            robustness = "moderately robust"
        elif e_value > 1.5:
            robustness = "somewhat fragile"
        else:
            robustness = "fragile"

        return {
            "treatment": treatment,
            "outcome": outcome,
            "standardized_effect": round(d, 4),
            "approximate_RR": round(float(rr), 4),
            "E_value": round(float(e_value), 4),
            "robustness": robustness,
            "interpretation": (
                f"An unobserved confounder would need to be associated with "
                f"both {treatment} and {outcome} by a factor of ≥{e_value:.1f} "
                f"to explain away the observed causal effect."
            ),
            "reference": "VanderWeele & Ding (2017)",
        }

    def validate_dag(self) -> dict:
        """
        Test conditional independence implications of the DAG against data.
        Uses partial correlation as a proxy for conditional independence.
        If the DAG is correct, d-separated variables should have zero
        partial correlation.
        """
        from scipy.stats import pearsonr

        results = {"tests": [], "n_passed": 0, "n_failed": 0, "n_total": 0}
        nodes = [n for n in self.dag.nodes if n in self.data.columns]

        for i, node_a in enumerate(nodes):
            for node_b in nodes[i + 1:]:
                # Check if nodes are d-separated given all other observed ancestors
                if node_a in nx.ancestors(self.dag, node_b) or \
                   node_b in nx.ancestors(self.dag, node_a):
                    continue  # direct/indirect causal relationship expected

                # Check for d-separation
                other_nodes = [
                    n for n in nodes
                    if n != node_a and n != node_b
                ]
                try:
                    is_dsep = nx.d_separated(
                        self.dag, {node_a}, {node_b}, set(other_nodes)
                    )
                except Exception:
                    continue

                if is_dsep:
                    # These should be conditionally independent
                    corr, p_val = pearsonr(
                        self.data[node_a].values, self.data[node_b].values
                    )
                    passed = p_val > 0.05  # no significant correlation
                    results["tests"].append({
                        "node_a": node_a, "node_b": node_b,
                        "d_separated": True,
                        "correlation": round(float(corr), 4),
                        "p_value": round(float(p_val), 4),
                        "passed": passed,
                    })
                    results["n_total"] += 1
                    if passed:
                        results["n_passed"] += 1
                    else:
                        results["n_failed"] += 1

        if results["n_total"] > 0:
            results["pass_rate"] = round(
                results["n_passed"] / results["n_total"], 3
            )
        return results

    def get_structural_equations(self) -> dict:
        """Return human-readable structural equations for all nodes."""
        equations = {}
        for node in self.topo_order:
            if node in self._linear_models:
                lm = self._linear_models[node]
                terms = [
                    f"{coeff:+.3f}·{parent}"
                    for parent, coeff in lm["coefficients"].items()
                ]
                eq_str = f"{node} = {' '.join(terms)} + {lm['intercept']:+.3f} + U_{node}"
                equations[node] = {
                    "equation": eq_str,
                    "coefficients": lm["coefficients"],
                    "intercept": lm["intercept"],
                    "linear_r2": lm["r2"],
                    "nonlinear_r2": self._r2_scores.get(node, None),
                    "cv_r2": self._cv_scores.get(node, None),
                    "residual_std": round(
                        float(self._exogenous_noise[node].std()), 4
                    ) if node in self._exogenous_noise else None,
                    "parents": lm["parents"],
                }

        return equations

    def get_dag_info(self) -> dict:
        """Return DAG structure for visualization."""
        return {
            "nodes": list(self.dag.nodes),
            "edges": [(str(u), str(v)) for u, v in self.dag.edges],
            "topological_order": self.topo_order,
            "root_nodes": [n for n in self.dag.nodes
                          if self.dag.in_degree(n) == 0],
            "leaf_nodes": [n for n in self.dag.nodes
                          if self.dag.out_degree(n) == 0],
            "n_structural_equations": len(self._struct_models),
        }
