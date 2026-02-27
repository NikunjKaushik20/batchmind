"""
First-Principles Pharmaceutical Manufacturing Physics Engine

Models implemented (with literature references):
  1. Heckel Equation — compression pressure → tablet relative density
     ln(1/(1-D)) = K·P + A   [Heckel, 1961]
  2. Ryshkewitch-Duckworth — porosity → tensile strength (≈ hardness proxy)
     σ = σ₀·exp(-b·ε)        [Ryshkewitch, 1953]
  3. Page Drying Kinetics — (temp, time, binder) → residual moisture
     M(t) = M₀·exp(-k·tⁿ), k = k₀·exp(-Eₐ/RT)  [Page, 1949]
  4. Power Consumption Model — (force, speed) → energy
     P = α·F·v + β·v² + P₀   [empirical]

All constants are calibrated from historical batch data via scipy.optimize.curve_fit.
Physics predictions serve as:
  - Independent validation of ML surrogate models
  - Constraints in NSGA-II optimization
  - Physically grounded latent variable estimation (Moisture_Content)
  - The "Physics" half of "Causal-Physics Hybrid Intelligence"
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")


class PharmPhysicsEngine:
    """
    Calibrated first-principles physics for pharmaceutical tablet manufacturing.
    Each model is fit from historical data, then used for:
      - Latent variable estimation (Moisture_Content via drying kinetics)
      - Physics-informed constraints in optimization
      - Cross-validation of ML surrogate predictions
    """

    def __init__(self):
        # Heckel parameters
        self.heckel_K = 0.015      # yield pressure reciprocal (1/MPa)
        self.heckel_A = 0.30       # densification at zero pressure

        # Ryshkewitch parameters
        self.rys_sigma0 = 150.0    # tensile strength at zero porosity (N)
        self.rys_b = 4.5           # porosity sensitivity

        # Page drying parameters
        self.page_k0 = 0.008       # pre-exponential factor
        self.page_Ea = 25000.0     # activation energy (J/mol)
        self.page_n = 1.15         # Page exponent
        self.page_M0_base = 4.2    # base initial moisture (%)
        self.page_binder_coeff = 0.35  # binder → initial moisture

        # Power model parameters
        self.power_alpha = 0.012   # force-speed coupling
        self.power_beta = 0.0008   # speed² term
        self.power_P0 = 1.5        # base power (kW)

        # Calibration metadata
        self._calibrated = False
        self._calibration_r2 = {}

    # ─── HECKEL EQUATION ──────────────────────────────────────────────────────

    @staticmethod
    def _heckel_func(P, K, A):
        """Heckel: D = 1 - exp(-(K·P + A))"""
        return 1.0 - np.exp(-(K * P + A))

    def heckel_density(self, pressure: float) -> float:
        """Predict tablet relative density from compression pressure."""
        return float(self._heckel_func(pressure, self.heckel_K, self.heckel_A))

    # ─── RYSHKEWITCH-DUCKWORTH ────────────────────────────────────────────────

    @staticmethod
    def _ryshkewitch_func(porosity, sigma0, b):
        """σ = σ₀·exp(-b·ε)"""
        return sigma0 * np.exp(-b * porosity)

    def ryshkewitch_strength(self, porosity: float) -> float:
        """Predict tensile strength from porosity (proxy for hardness)."""
        return float(self._ryshkewitch_func(porosity, self.rys_sigma0, self.rys_b))

    def predict_hardness(self, compression_force: float, lubricant_conc: float) -> float:
        """
        Physics chain: Compression_Force → density → porosity → hardness.
        Lubricant reduces inter-particle bonding (empirical correction).
        """
        density = self.heckel_density(compression_force)
        porosity = max(0.01, 1.0 - density)
        base_strength = self.ryshkewitch_strength(porosity)
        # Lubricant weakens bonds: empirical correction factor
        lub_factor = 1.0 - 0.15 * (lubricant_conc - 0.5)
        return float(base_strength * lub_factor)

    # ─── PAGE DRYING KINETICS ─────────────────────────────────────────────────

    def page_drying_moisture(self, drying_temp: float, drying_time: float,
                              binder_amount: float) -> float:
        """
        Page model with Arrhenius temperature dependence:
          M(t) = M₀·exp(-k·tⁿ)
          k = k₀·exp(-Eₐ/(R·T))
          M₀ = base + binder_coeff × binder_amount

        Returns estimated residual moisture content (%).
        This is the physics-based estimator for the latent variable Moisture_Content.
        """
        R = 8.314  # J/(mol·K)
        T_kelvin = drying_temp + 273.15

        # Initial moisture content: increases with binder amount
        M0 = self.page_M0_base + self.page_binder_coeff * binder_amount

        # Arrhenius rate constant
        k = self.page_k0 * np.exp(-self.page_Ea / (R * T_kelvin))

        # Page model
        M = M0 * np.exp(-k * drying_time ** self.page_n)
        return float(np.clip(M, 0.3, 8.0))

    # ─── POWER CONSUMPTION MODEL ─────────────────────────────────────────────

    @staticmethod
    def _power_func(X, alpha, beta, P0):
        """P = α·F·v + β·v² + P₀"""
        force, speed = X
        return alpha * force * speed + beta * speed ** 2 + P0

    def predict_power(self, compression_force: float, machine_speed: float) -> float:
        """Predict power consumption (kW) from force and speed."""
        return float(self._power_func(
            (compression_force, machine_speed),
            self.power_alpha, self.power_beta, self.power_P0
        ))

    # ─── CALIBRATION FROM DATA ────────────────────────────────────────────────

    def calibrate(self, df: pd.DataFrame) -> dict:
        """
        Calibrate all physics model parameters from historical batch data.
        Uses scipy.optimize.curve_fit with bounded optimization.
        Returns R² for each sub-model as a quality metric.
        """
        results = {}

        # 1. Calibrate Heckel + Ryshkewitch via Hardness prediction
        try:
            cf = df["Compression_Force"].values
            lc = df["Lubricant_Conc"].values
            hardness = df["Hardness"].values

            def combined_hardness(X, K, A, sigma0, b, lub_scale):
                force, lub = X
                density = 1.0 - np.exp(-(K * force + A))
                porosity = np.maximum(0.01, 1.0 - density)
                strength = sigma0 * np.exp(-b * porosity)
                return strength * (1.0 - lub_scale * (lub - 0.5))

            popt, _ = curve_fit(
                combined_hardness, (cf, lc), hardness,
                p0=[self.heckel_K, self.heckel_A, self.rys_sigma0, self.rys_b, 0.15],
                bounds=([0.001, 0.01, 10, 0.5, 0.0], [0.1, 2.0, 500, 15, 0.5]),
                maxfev=5000,
            )
            self.heckel_K, self.heckel_A = popt[0], popt[1]
            self.rys_sigma0, self.rys_b = popt[2], popt[3]

            pred = combined_hardness((cf, lc), *popt)
            ss_res = np.sum((hardness - pred) ** 2)
            ss_tot = np.sum((hardness - hardness.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            self._calibration_r2["hardness"] = round(r2, 4)
            results["hardness"] = {"r2": round(r2, 4), "params": dict(zip(
                ["K", "A", "sigma0", "b", "lub_scale"], [round(float(p), 6) for p in popt]
            ))}
        except Exception as e:
            results["hardness"] = {"error": str(e)}

        # 2. Calibrate Power model
        try:
            if "Power_kWh" in df.columns:
                power_col = "Power_kWh"
            elif "total_kwh" in df.columns:
                power_col = "total_kwh"
            else:
                power_col = None

            if power_col and power_col in df.columns:
                cf_vals = df["Compression_Force"].values
                ms_vals = df["Machine_Speed"].values
                power_vals = df[power_col].values

                popt, _ = curve_fit(
                    self._power_func, (cf_vals, ms_vals), power_vals,
                    p0=[self.power_alpha, self.power_beta, self.power_P0],
                    bounds=([0, 0, 0], [0.1, 0.01, 10]),
                    maxfev=5000,
                )
                self.power_alpha, self.power_beta, self.power_P0 = popt

                pred = self._power_func((cf_vals, ms_vals), *popt)
                ss_res = np.sum((power_vals - pred) ** 2)
                ss_tot = np.sum((power_vals - power_vals.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                self._calibration_r2["power"] = round(r2, 4)
                results["power"] = {"r2": round(r2, 4), "params": {
                    "alpha": round(float(popt[0]), 6),
                    "beta": round(float(popt[1]), 6),
                    "P0": round(float(popt[2]), 4),
                }}
        except Exception as e:
            results["power"] = {"error": str(e)}

        # 3. Page drying model — calibrate against Dissolution_Rate as a Moisture proxy
        # Higher moisture → lower dissolution; we calibrate the kinetics indirectly
        try:
            dt_vals = df["Drying_Temp"].values
            dtime_vals = df["Drying_Time"].values
            binder_vals = df["Binder_Amount"].values
            dissolution = df["Dissolution_Rate"].values

            def drying_dissolution_proxy(X, k0, Ea, n, M0_base, b_coeff, d_slope, d_intercept):
                temp, time, binder = X
                R = 8.314
                T_k = temp + 273.15
                M0 = M0_base + b_coeff * binder
                k = k0 * np.exp(-Ea / (R * T_k))
                moisture = M0 * np.exp(-k * time ** n)
                return d_intercept - d_slope * moisture

            popt, _ = curve_fit(
                drying_dissolution_proxy,
                (dt_vals, dtime_vals, binder_vals), dissolution,
                p0=[0.008, 25000, 1.15, 4.2, 0.35, 3.0, 95.0],
                bounds=([0.0001, 5000, 0.5, 1.0, 0.05, 0.1, 50],
                        [0.1, 80000, 3.0, 10.0, 2.0, 20, 110]),
                maxfev=10000,
            )
            self.page_k0 = popt[0]
            self.page_Ea = popt[1]
            self.page_n = popt[2]
            self.page_M0_base = popt[3]
            self.page_binder_coeff = popt[4]

            pred = drying_dissolution_proxy((dt_vals, dtime_vals, binder_vals), *popt)
            ss_res = np.sum((dissolution - pred) ** 2)
            ss_tot = np.sum((dissolution - dissolution.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            self._calibration_r2["drying"] = round(r2, 4)
            results["drying_kinetics"] = {"r2": round(r2, 4), "params": {
                "k0": round(float(popt[0]), 6), "Ea": round(float(popt[1]), 1),
                "n": round(float(popt[2]), 4), "M0_base": round(float(popt[3]), 4),
                "binder_coeff": round(float(popt[4]), 4),
            }}
        except Exception as e:
            results["drying_kinetics"] = {"error": str(e)}

        self._calibrated = True
        return results

    # ─── CONSTRAINT GENERATION FOR NSGA-II ────────────────────────────────────

    def get_manufacturing_constraints(self) -> dict:
        """
        Return pharmaceutical manufacturing constraints for NSGA-II.
        Based on USP/EP pharmacopoeial standards for oral solid dosage forms.
        """
        return {
            "Hardness": {"min": 40.0, "max": 120.0, "unit": "N"},
            "Friability": {"max": 1.0, "unit": "%"},
            "Dissolution_Rate": {"min": 80.0, "unit": "%"},
            "Disintegration_Time": {"max": 15.0, "unit": "min"},
            "Content_Uniformity": {"min": 85.0, "max": 115.0, "unit": "%"},
            "Moisture_Content": {"max": 5.0, "unit": "%"},
        }

    # ─── FULL PREDICTION PIPELINE ─────────────────────────────────────────────

    def predict_all(self, params: dict) -> dict:
        """
        Physics-based prediction of manufacturing outcomes from input parameters.
        Returns point estimates for all physics-modeled outputs.
        """
        cf = params.get("Compression_Force", 15)
        ms = params.get("Machine_Speed", 30)
        dt = params.get("Drying_Temp", 60)
        dtime = params.get("Drying_Time", 45)
        ba = params.get("Binder_Amount", 5)
        lc = params.get("Lubricant_Conc", 0.5)

        moisture = self.page_drying_moisture(dt, dtime, ba)
        hardness = self.predict_hardness(cf, lc)
        power_kw = self.predict_power(cf, ms)

        return {
            "Moisture_Content": round(moisture, 3),
            "Hardness_physics": round(hardness, 3),
            "Power_kW_physics": round(power_kw, 3),
            "model": "first-principles",
            "calibrated": self._calibrated,
            "calibration_r2": self._calibration_r2,
        }

    def get_info(self) -> dict:
        """Return physics model equations and parameters for display."""
        return {
            "models": {
                "Heckel": {
                    "equation": "D = 1 - exp(-(K·P + A))",
                    "reference": "Heckel, 1961",
                    "params": {"K": self.heckel_K, "A": self.heckel_A},
                },
                "Ryshkewitch": {
                    "equation": "σ = σ₀·exp(-b·ε)",
                    "reference": "Ryshkewitch, 1953",
                    "params": {"sigma0": self.rys_sigma0, "b": self.rys_b},
                },
                "Page_Drying": {
                    "equation": "M(t) = M₀·exp(-k·tⁿ), k = k₀·exp(-Eₐ/RT)",
                    "reference": "Page, 1949",
                    "params": {
                        "k0": self.page_k0, "Ea": self.page_Ea,
                        "n": self.page_n, "M0_base": self.page_M0_base,
                    },
                },
                "Power": {
                    "equation": "P = α·F·v + β·v² + P₀",
                    "reference": "Empirical",
                    "params": {
                        "alpha": self.power_alpha, "beta": self.power_beta,
                        "P0": self.power_P0,
                    },
                },
            },
            "calibrated": self._calibrated,
            "calibration_r2": self._calibration_r2,
            "constraints": self.get_manufacturing_constraints(),
        }


# Module-level singleton
_physics_engine = None


def get_physics_engine() -> PharmPhysicsEngine:
    global _physics_engine
    if _physics_engine is None:
        _physics_engine = PharmPhysicsEngine()
    return _physics_engine
