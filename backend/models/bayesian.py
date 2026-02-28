"""
Bayesian Inference Engine — Real Conjugate Posterior Updating

Implements Normal-Inverse-Gamma (NIG) conjugate family for Bayesian
inference on manufacturing parameters with unknown mean and variance.

Mathematics:
  Prior:     (μ, σ²) ~ NIG(μ₀, κ₀, α₀, β₀)
  Marginal:  μ ~ t_{2α₀}(μ₀, β₀/(α₀·κ₀))
  Posterior after n observations with sample mean x̄, sample var s²:
    κₙ = κ₀ + n
    μₙ = (κ₀·μ₀ + n·x̄) / κₙ
    αₙ = α₀ + n/2
    βₙ = β₀ + ½·Σ(xᵢ-x̄)² + κ₀·n·(x̄-μ₀)² / (2·κₙ)
  Predictive: x_new ~ t_{2αₙ}(μₙ, βₙ·(κₙ+1)/(αₙ·κₙ))
  Credible interval: from posterior marginal t-distribution

References:
  Murphy, K. P. (2007). Conjugate Bayesian analysis of the Gaussian distribution.
  Gelman et al. (2013). Bayesian Data Analysis, 3rd ed., Chapter 3.
"""

import numpy as np
import logging
from scipy.stats import t as t_dist, invgamma, norm
from scipy.special import gammaln
import json

logger = logging.getLogger(__name__)


class NormalInverseGamma:
    """
    Conjugate Bayesian model for Normal data with unknown mean and variance.

    The NIG distribution is the natural conjugate prior for the Normal
    distribution when both μ and σ² are unknown.

    Parameters:
        mu0    : prior mean
        kappa0 : prior pseudo-observations for mean (strength of belief)
        alpha0 : shape parameter for inverse-gamma on variance
        beta0  : rate parameter for inverse-gamma on variance
    """

    def __init__(self, mu0: float, kappa0: float, alpha0: float, beta0: float):
        self.mu = float(mu0)
        self.kappa = float(max(kappa0, 0.01))
        self.alpha = float(max(alpha0, 0.51))  # must be > 0.5 for finite variance
        self.beta = float(max(beta0, 0.01))
        self.n_updates = 0
        self._update_history = []
        # Store original prior for Savage-Dickey Bayes Factor
        self._prior_mu0 = self.mu
        self._prior_kappa0 = self.kappa
        self._prior_alpha0 = self.alpha
        self._prior_beta0 = self.beta

    def update(self, data: np.ndarray) -> dict:
        """
        Bayesian posterior update given new observations.

        Implements the exact NIG conjugate update equations:
          κₙ = κ₀ + n
          μₙ = (κ₀·μ₀ + n·x̄) / κₙ
          αₙ = α₀ + n/2
          βₙ = β₀ + ½·(n-1)·s² + κ₀·n·(x̄-μ₀)² / (2·κₙ)

        Returns dict with prior → posterior parameter changes.
        """
        data = np.asarray(data, dtype=float).flatten()
        n = len(data)
        if n == 0:
            return {"updated": False, "reason": "no data"}

        x_bar = np.mean(data)
        s2 = np.var(data, ddof=0) if n > 1 else 0.0

        # Store prior state
        prior_state = self.get_state()

        # NIG conjugate update
        kappa_n = self.kappa + n
        mu_n = (self.kappa * self.mu + n * x_bar) / kappa_n
        alpha_n = self.alpha + n / 2.0
        beta_n = (self.beta
                  + 0.5 * n * s2
                  + (self.kappa * n * (x_bar - self.mu) ** 2) / (2.0 * kappa_n))

        # Update parameters
        self.kappa = kappa_n
        self.mu = mu_n
        self.alpha = alpha_n
        self.beta = beta_n
        self.n_updates += 1

        posterior_state = self.get_state()
        self._update_history.append({
            "n_obs": n, "x_bar": round(x_bar, 4),
            "prior_mu": prior_state["mu"], "posterior_mu": posterior_state["mu"],
        })

        return {
            "updated": True,
            "n_observations": n,
            "prior": prior_state,
            "posterior": posterior_state,
        }

    def posterior_mean(self) -> float:
        """E[μ | data] = μₙ"""
        return self.mu

    def posterior_variance(self) -> float:
        """Var[μ | data] = βₙ / ((αₙ - 1) · κₙ) for αₙ > 1"""
        if self.alpha > 1:
            return self.beta / ((self.alpha - 1) * self.kappa)
        return float("inf")

    def expected_data_variance(self) -> float:
        """E[σ² | data] = βₙ / (αₙ - 1) for αₙ > 1"""
        if self.alpha > 1:
            return self.beta / (self.alpha - 1)
        return float("inf")

    def credible_interval(self, level: float = 0.95) -> tuple:
        """
        Bayesian credible interval for the mean μ from the posterior
        marginal t-distribution:
          μ | data ~ t_{2αₙ}(μₙ, βₙ/(αₙ·κₙ))

        This is a CREDIBLE interval (Bayesian), not a confidence interval (frequentist).
        The interpretation: P(μ ∈ [lo, hi] | data) = level.
        """
        df = 2.0 * self.alpha
        scale = np.sqrt(self.beta / (self.alpha * self.kappa))
        lo = t_dist.ppf((1 - level) / 2, df, loc=self.mu, scale=scale)
        hi = t_dist.ppf(1 - (1 - level) / 2, df, loc=self.mu, scale=scale)
        return (float(lo), float(hi))

    def predictive_credible_interval(self, level: float = 0.95) -> tuple:
        """
        Posterior predictive interval for the NEXT observation x_{n+1}:
          x_{n+1} | data ~ t_{2αₙ}(μₙ, βₙ·(κₙ+1)/(αₙ·κₙ))

        This accounts for both parameter uncertainty AND data noise.
        """
        df = 2.0 * self.alpha
        scale = np.sqrt(self.beta * (self.kappa + 1) / (self.alpha * self.kappa))
        lo = t_dist.ppf((1 - level) / 2, df, loc=self.mu, scale=scale)
        hi = t_dist.ppf(1 - (1 - level) / 2, df, loc=self.mu, scale=scale)
        return (float(lo), float(hi))

    def posterior_predictive_pdf(self, x: float) -> float:
        """
        Posterior predictive density at point x:
          p(x | data) = t_{2αₙ}(x; μₙ, βₙ·(κₙ+1)/(αₙ·κₙ))
        """
        df = 2.0 * self.alpha
        scale = np.sqrt(self.beta * (self.kappa + 1) / (self.alpha * self.kappa))
        return float(t_dist.pdf(x, df, loc=self.mu, scale=scale))

    def log_marginal_likelihood(self, data: np.ndarray) -> float:
        """
        Log marginal likelihood (evidence) for Bayesian model comparison:
          log p(x₁:ₙ) = -n/2·log(π) + ½·log(κ₀/κₙ) + α₀·log(β₀) - αₙ·log(βₙ)
                        + log Γ(αₙ) - log Γ(α₀)

        Used for Bayes Factor computation between competing signatures.
        """
        data = np.asarray(data, dtype=float).flatten()
        n = len(data)
        if n == 0:
            return 0.0

        x_bar = np.mean(data)
        s2 = np.var(data, ddof=0) if n > 1 else 0.0

        kappa_n = self.kappa + n
        alpha_n = self.alpha + n / 2.0
        beta_n = (self.beta + 0.5 * n * s2
                  + (self.kappa * n * (x_bar - self.mu) ** 2) / (2.0 * kappa_n))

        log_ml = (
            -0.5 * n * np.log(np.pi)
            + 0.5 * np.log(self.kappa / kappa_n)
            + self.alpha * np.log(self.beta) - alpha_n * np.log(beta_n)
            + gammaln(alpha_n) - gammaln(self.alpha)
        )
        return float(log_ml)

    def get_state(self) -> dict:
        """Snapshot of current posterior state."""
        ci = self.credible_interval(0.95)
        pred_ci = self.predictive_credible_interval(0.95)
        return {
            "mu": round(self.mu, 4),
            "kappa": round(self.kappa, 4),
            "alpha": round(self.alpha, 4),
            "beta": round(self.beta, 4),
            "posterior_mean": round(self.posterior_mean(), 4),
            "posterior_variance": round(self.posterior_variance(), 6)
                if self.posterior_variance() < 1e6 else "inf",
            "credible_interval_95": [round(ci[0], 4), round(ci[1], 4)],
            "predictive_interval_95": [round(pred_ci[0], 4), round(pred_ci[1], 4)],
            "n_updates": self.n_updates,
        }


class BayesianParameterTracker:
    """
    Multi-parameter Bayesian tracker for golden signature parameters.

    Maintains independent NIG posteriors for each manufacturing parameter.
    Updates all posteriors when a new qualifying batch is observed.
    Supports Bayesian model comparison between competing signatures.
    """

    def __init__(self, param_names: list, initial_data: dict = None):
        """
        Initialize posteriors for each parameter.

        Args:
            param_names: list of parameter names
            initial_data: dict of {param_name: array of initial values}
                          Used to set informative priors from historical data.
        """
        self.param_names = param_names
        self.posteriors = {}
        self.total_batches_seen = 0

        for param in param_names:
            if initial_data and param in initial_data:
                values = np.asarray(initial_data[param], dtype=float)
                mu0 = float(np.mean(values))
                # Weak prior: κ₀ = 1 (equivalent to 1 pseudo-observation)
                kappa0 = 1.0
                # α₀ = 2 gives finite variance, not too informative
                alpha0 = 2.0
                # β₀ set from data variance: E[σ²] = β₀/(α₀-1) ≈ observed variance
                var_est = float(np.var(values)) + 1e-6
                beta0 = var_est * (alpha0 - 1)

                nig = NormalInverseGamma(mu0, kappa0, alpha0, beta0)
                # Do initial update with the data to form posterior
                nig.update(values)
                self.posteriors[param] = nig
            else:
                # Uninformative prior
                self.posteriors[param] = NormalInverseGamma(0.0, 0.01, 1.01, 1.0)

    def update_from_batch(self, batch_params: dict) -> dict:
        """
        Update posteriors using parameter values from a new qualifying batch.

        Args:
            batch_params: dict of {param_name: observed_value}

        Returns:
            dict with update results for each parameter.
        """
        results = {}
        for param in self.param_names:
            if param in batch_params:
                value = float(batch_params[param])
                result = self.posteriors[param].update(np.array([value]))
                results[param] = result
        self.total_batches_seen += 1
        return {"updated_params": results, "total_batches_seen": self.total_batches_seen}

    def get_credible_intervals(self, level: float = 0.95) -> dict:
        """
        Get Bayesian credible intervals for all parameters.

        These are TRUE Bayesian credible intervals from the posterior
        t-distribution. NOT frequentist confidence intervals.

        Interpretation: P(μ_param ∈ [lo, hi] | all observed data) = level
        """
        intervals = {}
        for param in self.param_names:
            ci = self.posteriors[param].credible_interval(level)
            pred_ci = self.posteriors[param].predictive_credible_interval(level)
            intervals[param] = {
                "value": round(self.posteriors[param].posterior_mean(), 3),
                "ci_low": round(ci[0], 3),
                "ci_high": round(ci[1], 3),
                "predictive_low": round(pred_ci[0], 3),
                "predictive_high": round(pred_ci[1], 3),
                "posterior_variance": round(self.posteriors[param].posterior_variance(), 6)
                    if self.posteriors[param].posterior_variance() < 1e6 else "inf",
                "std": round(np.sqrt(self.posteriors[param].posterior_variance()), 4)
                    if self.posteriors[param].posterior_variance() < 1e6 else "inf",
                "n_updates": self.posteriors[param].n_updates,
                "type": "bayesian_credible",
            }
        return intervals

    def get_posterior_state(self) -> dict:
        """Full posterior state for all parameters (for API/visualization)."""
        state = {}
        for param in self.param_names:
            state[param] = self.posteriors[param].get_state()
        return {
            "parameters": state,
            "total_batches_seen": self.total_batches_seen,
            "inference_type": "Normal-Inverse-Gamma conjugate",
        }

    def bayes_factor(self, param: str, hypothesis_value: float,
                     data: np.ndarray = None) -> dict:
        """
        Compute Bayes Factor for testing H₀: μ = hypothesis_value
        vs H₁: μ ≠ hypothesis_value (Savage-Dickey density ratio).

        BF₀₁ = p(μ = h | data) / p(μ = h | prior)

        Prior marginal:     μ ~ t_{2α₀}(μ₀, β₀/(α₀·κ₀))
        Posterior marginal: μ ~ t_{2αₙ}(μₙ, βₙ/(αₙ·κₙ))

        BF > 3: substantial evidence for H₀
        BF < 1/3: substantial evidence against H₀
        """
        if param not in self.posteriors:
            return {"error": f"Parameter {param} not tracked"}

        nig = self.posteriors[param]

        # Posterior marginal density: μ | data ~ t_{2αₙ}(μₙ, βₙ/(αₙ·κₙ))
        df_post = 2.0 * nig.alpha
        scale_post = np.sqrt(nig.beta / (nig.alpha * nig.kappa))
        posterior_density = t_dist.pdf(hypothesis_value, df_post,
                                       loc=nig.mu, scale=scale_post)

        # TRUE prior marginal density: μ ~ t_{2α₀}(μ₀, β₀/(α₀·κ₀))
        df_prior = 2.0 * nig._prior_alpha0
        scale_prior = np.sqrt(nig._prior_beta0 / (nig._prior_alpha0 * nig._prior_kappa0))
        prior_density = t_dist.pdf(hypothesis_value, df_prior,
                                    loc=nig._prior_mu0, scale=scale_prior)
        prior_density = max(prior_density, 1e-30)

        bf = posterior_density / prior_density
        bf = max(bf, 1e-30)  # prevent log(0)

        if bf > 10:
            interpretation = "strong evidence FOR hypothesis"
        elif bf > 3:
            interpretation = "moderate evidence FOR hypothesis"
        elif bf > 1 / 3:
            interpretation = "inconclusive"
        elif bf > 1 / 10:
            interpretation = "moderate evidence AGAINST hypothesis"
        else:
            interpretation = "strong evidence AGAINST hypothesis"

        return {
            "bayes_factor": round(float(bf), 4),
            "log_bf": round(float(np.log(bf)), 4),
            "interpretation": interpretation,
            "hypothesis_value": hypothesis_value,
            "prior_params": {
                "mu0": nig._prior_mu0, "kappa0": nig._prior_kappa0,
                "alpha0": nig._prior_alpha0, "beta0": nig._prior_beta0,
            },
            "posterior_params": {
                "mu": nig.mu, "kappa": nig.kappa,
                "alpha": nig.alpha, "beta": nig.beta,
            },
        }
