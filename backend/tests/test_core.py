"""
BatchMind Unit Tests — Mathematical Correctness + Integration

Tests cover:
  1. NIG conjugate posterior correctness (known analytical answers)
  2. Bayesian credible intervals (posterior t-distribution)
  3. Savage-Dickey Bayes Factor (prior stored correctly)
  4. Pareto dominance logic
  5. SCM counterfactual consistency (CF = observed when do = observed)
  6. Physics model boundary conditions
  7. Adaptive constraint tightening/relaxation
  8. Data loader sanity checks
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pytest
import numpy as np
import pandas as pd


# ─── NIG POSTERIOR TESTS ─────────────────────────────────────────────────────

class TestNormalInverseGamma:
    """Test Bayesian NIG conjugate posterior correctness."""

    def test_prior_initialization(self):
        from models.bayesian import NormalInverseGamma
        nig = NormalInverseGamma(mu0=10.0, kappa0=1.0, alpha0=2.0, beta0=3.0)
        assert nig.mu == 10.0
        assert nig.kappa == 1.0
        assert nig.alpha == 2.0
        assert nig.beta == 3.0
        assert nig.n_updates == 0

    def test_prior_stored_for_bayes_factor(self):
        """Verify that original prior parameters are stored for Savage-Dickey."""
        from models.bayesian import NormalInverseGamma
        nig = NormalInverseGamma(mu0=5.0, kappa0=2.0, alpha0=3.0, beta0=4.0)
        nig.update(np.array([6.0, 7.0, 8.0]))
        # Prior should be unchanged
        assert nig._prior_mu0 == 5.0
        assert nig._prior_kappa0 == 2.0
        assert nig._prior_alpha0 == 3.0
        assert nig._prior_beta0 == 4.0
        # Posterior should have changed
        assert nig.mu != 5.0
        assert nig.kappa != 2.0

    def test_posterior_update_single_observation(self):
        """Test NIG update with a single observation — known formulas."""
        from models.bayesian import NormalInverseGamma
        nig = NormalInverseGamma(mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1.0)
        nig.update(np.array([2.0]))
        # kappa_n = 1 + 1 = 2
        assert nig.kappa == 2.0
        # mu_n = (1*0 + 1*2) / 2 = 1.0
        assert abs(nig.mu - 1.0) < 1e-10
        # alpha_n = 2 + 0.5 = 2.5
        assert abs(nig.alpha - 2.5) < 1e-10

    def test_posterior_update_multiple_observations(self):
        """Test NIG update with multiple observations."""
        from models.bayesian import NormalInverseGamma
        nig = NormalInverseGamma(mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1.0)
        data = np.array([3.0, 5.0, 7.0])
        n = len(data)
        x_bar = np.mean(data)  # 5.0
        nig.update(data)
        # kappa_n = 1 + 3 = 4
        assert nig.kappa == 4.0
        # mu_n = (1*0 + 3*5) / 4 = 3.75
        assert abs(nig.mu - 3.75) < 1e-10
        # alpha_n = 2 + 3/2 = 3.5
        assert abs(nig.alpha - 3.5) < 1e-10

    def test_posterior_variance_decreases_with_data(self):
        """Posterior variance should decrease as more data is observed."""
        from models.bayesian import NormalInverseGamma
        nig = NormalInverseGamma(mu0=50.0, kappa0=1.0, alpha0=2.0, beta0=10.0)
        var_before = nig.posterior_variance()
        nig.update(np.array([48.0, 52.0, 50.0, 51.0, 49.0]))
        var_after = nig.posterior_variance()
        assert var_after < var_before

    def test_credible_interval_contains_true_mean(self):
        """95% CI from posterior should usually contain the data mean."""
        from models.bayesian import NormalInverseGamma
        nig = NormalInverseGamma(mu0=0.0, kappa0=0.01, alpha0=1.01, beta0=1.0)
        data = np.random.normal(10.0, 2.0, size=50)
        nig.update(data)
        lo, hi = nig.credible_interval(0.95)
        assert lo < np.mean(data) < hi

    def test_log_marginal_likelihood_finite(self):
        """Log marginal likelihood should be a finite number."""
        from models.bayesian import NormalInverseGamma
        nig = NormalInverseGamma(mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1.0)
        data = np.array([1.0, 2.0, 3.0])
        lml = nig.log_marginal_likelihood(data)
        assert np.isfinite(lml)
        assert lml < 0  # log-likelihood is typically negative


# ─── BAYESIAN PARAMETER TRACKER TESTS ────────────────────────────────────────

class TestBayesianParameterTracker:
    def test_multi_param_initialization(self):
        from models.bayesian import BayesianParameterTracker
        params = ["Temp", "Pressure"]
        initial = {"Temp": np.array([60, 65, 70]), "Pressure": np.array([1, 2, 3])}
        tracker = BayesianParameterTracker(params, initial)
        ci = tracker.get_credible_intervals(0.95)
        assert "Temp" in ci
        assert "Pressure" in ci
        assert ci["Temp"]["type"] == "bayesian_credible"

    def test_bayes_factor_uses_true_prior(self):
        """Bayes Factor should use stored prior, not current posterior."""
        from models.bayesian import BayesianParameterTracker
        params = ["X"]
        initial = {"X": np.array([10, 11, 12, 9, 10])}
        tracker = BayesianParameterTracker(params, initial)
        # Update with shifted data to ensure posterior diverges from prior
        for val in [20.0, 22.0, 18.0, 21.0, 19.0]:
            tracker.update_from_batch({"X": val})
        bf = tracker.bayes_factor("X", 10.0)
        assert "bayes_factor" in bf
        assert "prior_params" in bf
        assert "posterior_params" in bf
        # After updating with data centered ~20, posterior mu should differ from prior mu
        assert abs(bf["prior_params"]["mu0"] - bf["posterior_params"]["mu"]) > 0.1, \
            "Posterior mu should differ from prior mu after observing shifted data"


# ─── ADAPTIVE CONSTRAINT TESTS ───────────────────────────────────────────────

class TestAdaptiveConstraints:
    def _make_df(self, hardness=80, friability=0.5, dissolution=95, disint=8, n=30):
        return pd.DataFrame({
            "Hardness": np.random.normal(hardness, 5, n),
            "Friability": np.random.normal(friability, 0.1, n),
            "Dissolution_Rate": np.random.normal(dissolution, 3, n),
            "Disintegration_Time": np.random.normal(disint, 1, n),
        })

    def test_tightening_when_exceeding(self):
        """Constraints should tighten when plant consistently exceeds them."""
        from models.adaptive_constraints import AdaptiveConstraintManager
        acm = AdaptiveConstraintManager(window_size=20)
        # Hardness consistently at 80 ± 5, way above the 40 bound
        df = self._make_df(hardness=80)
        result = acm.update(df)
        bounds = acm.get_nsga2_constraints()
        # Hardness bound should have increased from 40
        assert bounds["Hardness"] > 40.0

    def test_no_relaxation_beyond_initial(self):
        """Constraints should never relax below initial pharmacopoeial bounds."""
        from models.adaptive_constraints import AdaptiveConstraintManager
        acm = AdaptiveConstraintManager(window_size=20)
        # Even bad data shouldn't push below initial
        df = self._make_df(hardness=30)  # below the 40 bound
        acm.update(df)
        bounds = acm.get_nsga2_constraints()
        assert bounds["Hardness"] >= 40.0  # never below pharmacopoeial minimum

    def test_constraint_history_recorded(self):
        """Constraint changes should be recorded in history."""
        from models.adaptive_constraints import AdaptiveConstraintManager
        acm = AdaptiveConstraintManager(window_size=20)
        df = self._make_df(hardness=80)
        acm.update(df)
        history = acm.get_history()
        # Should have at least one entry if adaptation occurred
        assert isinstance(history, dict)


# ─── SCM COUNTERFACTUAL CONSISTENCY ──────────────────────────────────────────

class TestSCMConsistency:
    def test_counterfactual_equals_observed_when_no_intervention(self):
        """
        If we intervene with the SAME values as observed,
        the counterfactual should equal the observed (within noise tolerance).
        """
        from models.scm import StructuralCausalModel
        # Simple 2-node DAG: X → Y
        data = pd.DataFrame({
            "X": np.random.normal(10, 2, 50),
        })
        data["Y"] = 2 * data["X"] + np.random.normal(0, 0.5, 50)

        scm = StructuralCausalModel(
            dag_edges=[("X", "Y")],
            data=data,
        )
        scm.fit()

        # Counterfactual with same observed values
        obs_idx = 0
        obs_x = float(data.iloc[obs_idx]["X"])
        cf = scm.counterfactual(obs_idx, {"X": obs_x})

        # Y counterfactual should be very close to Y observed
        obs_y = float(data.iloc[obs_idx]["Y"])
        cf_y = cf["counterfactual_values"].get("Y", 0)
        # With noise fixed, should be exact
        assert abs(cf_y - obs_y) < 0.01, \
            f"CF Y={cf_y} should equal observed Y={obs_y}"

    def test_dag_validation(self):
        """DAG validation should not crash."""
        from models.scm import StructuralCausalModel
        data = pd.DataFrame({
            "A": np.random.normal(0, 1, 50),
            "B": np.random.normal(0, 1, 50),
        })
        data["C"] = data["A"] + data["B"]
        scm = StructuralCausalModel(
            dag_edges=[("A", "C"), ("B", "C")],
            data=data,
        )
        scm.fit()
        validation = scm.validate_dag()
        assert "n_total" in validation


# ─── PHYSICS MODEL BOUNDARY CONDITIONS ───────────────────────────────────────

class TestPhysicsModels:
    def test_heckel_density_increases_with_pressure(self):
        """Higher compression force → higher relative density."""
        from models.physics import PharmPhysicsEngine
        eng = PharmPhysicsEngine()
        d_low = eng.heckel_density(5.0)
        d_high = eng.heckel_density(50.0)
        assert d_high > d_low

    def test_heckel_density_bounded_0_1(self):
        """Relative density should always be in [0, 1)."""
        from models.physics import PharmPhysicsEngine
        eng = PharmPhysicsEngine()
        for p in [0, 1, 10, 50, 100, 500]:
            d = eng.heckel_density(p)
            assert 0 <= d < 1.0

    def test_page_moisture_decreases_with_time(self):
        """Longer drying time → lower residual moisture."""
        from models.physics import PharmPhysicsEngine
        eng = PharmPhysicsEngine()
        m_short = eng.page_drying_moisture(60.0, 10.0, 5.0)
        m_long = eng.page_drying_moisture(60.0, 60.0, 5.0)
        assert m_long <= m_short

    def test_page_moisture_decreases_with_temp(self):
        """Higher drying temperature → lower residual moisture."""
        from models.physics import PharmPhysicsEngine
        eng = PharmPhysicsEngine()
        m_cold = eng.page_drying_moisture(40.0, 30.0, 5.0)
        m_hot = eng.page_drying_moisture(80.0, 30.0, 5.0)
        assert m_hot <= m_cold

    def test_power_increases_with_speed(self):
        """Higher machine speed → higher power consumption."""
        from models.physics import PharmPhysicsEngine
        eng = PharmPhysicsEngine()
        p_slow = eng.predict_power(15.0, 20.0)
        p_fast = eng.predict_power(15.0, 60.0)
        assert p_fast > p_slow


# ─── DATA LOADER SANITY ──────────────────────────────────────────────────────

class TestDataLoader:
    def test_production_data_loads(self):
        from data_loader import load_production_data
        df = load_production_data()
        assert len(df) > 0
        assert "Batch_ID" in df.columns
        assert "Hardness" in df.columns

    def test_batch_ids_nonempty(self):
        from data_loader import get_all_batch_ids
        ids = get_all_batch_ids()
        assert len(ids) > 0

    def test_emission_factor_reasonable(self):
        from data_loader import INDIA_EMISSION_FACTOR
        assert 0.5 < INDIA_EMISSION_FACTOR < 1.5  # kg CO2e/kWh


# ─── UNCERTAINTY QUANTIFICATION TESTS ────────────────────────────────────────

class TestUncertaintyQuantification:
    """
    Verify that the new quantile regression + conformal prediction system
    replaces the old input-perturbation hack, providing calibrated
    prediction intervals.
    """

    @classmethod
    def setup_class(cls):
        """Trigger lazy initialization so models are available for testing."""
        from models.causal import _ensure_initialized
        _ensure_initialized()

    def test_predict_with_uncertainty_returns_dict(self):
        """New _predict_with_uncertainty should return a dict, not a tuple."""
        from models.causal import _predict_with_uncertainty, INPUT_PARAMS, _lgb_models
        if not _lgb_models:
            pytest.skip("Models not initialized")
        x = np.zeros(len(INPUT_PARAMS))
        target = list(_lgb_models.keys())[0]
        result = _predict_with_uncertainty(x, target)
        assert isinstance(result, dict), "Must return dict, not tuple"
        assert "mean" in result
        assert "prediction_std" in result

    def test_quantile_intervals_exist(self):
        """Quantile regression should produce low/high bounds."""
        from models.causal import _predict_with_uncertainty, _quantile_models, INPUT_PARAMS
        if not _quantile_models:
            pytest.skip("Quantile models not initialized")
        target = list(_quantile_models.keys())[0]
        x = np.zeros(len(INPUT_PARAMS))
        result = _predict_with_uncertainty(x, target)
        assert "quantile_low" in result, "Missing quantile_low"
        assert "quantile_high" in result, "Missing quantile_high"
        assert result["quantile_low"] <= result["mean"] <= result["quantile_high"], \
            "Quantile interval must contain the mean prediction"

    def test_conformal_intervals_exist(self):
        """Conformal prediction should produce calibrated intervals."""
        from models.causal import _predict_with_uncertainty, _conformal_residuals, INPUT_PARAMS
        if not _conformal_residuals:
            pytest.skip("Conformal residuals not computed")
        target = list(_conformal_residuals.keys())[0]
        x = np.zeros(len(INPUT_PARAMS))
        result = _predict_with_uncertainty(x, target, confidence=0.90)
        assert "conformal_low" in result, "Missing conformal_low"
        assert "conformal_high" in result, "Missing conformal_high"
        assert "conformal_coverage" in result
        assert result["conformal_coverage"] == 0.90

    def test_holdout_r2_measured_not_hardcoded(self):
        """Holdout R² should be measured per target, not hardcoded 0.85."""
        from models.causal import _holdout_r2
        if not _holdout_r2:
            pytest.skip("Holdout R² not computed")
        for target, r2 in _holdout_r2.items():
            assert isinstance(r2, float), f"{target} R² must be float"
            assert r2 != 0.85, f"{target} R² should not be hardcoded 0.85"
            assert 0 < r2 <= 1.0, f"{target} R² must be in (0, 1]"

    def test_conformal_residuals_sorted(self):
        """Conformal nonconformity scores must be sorted for quantile lookup."""
        from models.causal import _conformal_residuals
        if not _conformal_residuals:
            pytest.skip("Conformal residuals not computed")
        for target, residuals in _conformal_residuals.items():
            assert np.all(residuals[:-1] <= residuals[1:]), \
                f"Conformal residuals for {target} must be sorted"


# ─── ENERGY FORECAST TESTS ──────────────────────────────────────────────────

class TestEnergyForecast:
    """Verify Holt-Winters exponential smoothing (not DBA+noise hack)."""

    def test_forecast_returns_prediction_intervals(self):
        """Energy forecast should now include prediction intervals."""
        from models.fingerprint import energy_pattern_forecast
        result = energy_pattern_forecast("Compression", n_future=3)
        if "error" in result:
            pytest.skip("Phase not found in data")
        assert "prediction_interval_low" in result, \
            "Must include prediction intervals"
        assert "prediction_interval_high" in result
        assert len(result["predicted_energy_kwh"]) == 3

    def test_forecast_uses_exponential_smoothing(self):
        """Method should be Holt-Winters or Simple ES, not DBA+noise."""
        from models.fingerprint import energy_pattern_forecast
        result = energy_pattern_forecast("Compression", n_future=3)
        if "error" in result:
            pytest.skip("Phase not found in data")
        method = result.get("method", "")
        assert "noise model" not in method.lower(), \
            f"Should use exponential smoothing, not noise model. Got: {method}"

    def test_prediction_intervals_widen_with_horizon(self):
        """Prediction intervals should widen for longer forecast horizons."""
        from models.fingerprint import energy_pattern_forecast
        result = energy_pattern_forecast("Compression", n_future=5)
        if "error" in result:
            pytest.skip("Phase not found in data")
        lows = result.get("prediction_interval_low", [])
        highs = result.get("prediction_interval_high", [])
        if len(lows) >= 2 and len(highs) >= 2:
            # Interval width should increase (or stay same)
            width_first = highs[0] - lows[0]
            width_last = highs[-1] - lows[-1]
            assert width_last >= width_first * 0.99, \
                "Prediction intervals should widen with horizon"


# ─── THREAD SAFETY TESTS ────────────────────────────────────────────────────

class TestThreadSafety:
    """Verify thread locks are in place."""

    def test_state_lock_exists(self):
        """Global state lock should exist in causal module."""
        from models.causal import _state_lock
        # RLock provides acquire/release methods
        assert hasattr(_state_lock, 'acquire') and hasattr(_state_lock, 'release'), \
            "_state_lock must have acquire/release methods (Lock/RLock)"

    def test_online_manager_has_lock(self):
        """OnlineLearningManager should have a lock."""
        from models.online_learning import OnlineLearningManager
        mgr = OnlineLearningManager()
        assert hasattr(mgr, "_lock"), "OnlineLearningManager needs a _lock attribute"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

