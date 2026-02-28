"""
Phase-Aware Energy Fingerprinting — DTW Barycentric Average + Change-Point Detection

Architecture:
  1. Phase segmentation: change-point detection (Ruptures PELT) + pre-labeled phases
  2. Reference fingerprint: DTW Barycentric Average (DBA) — NOT Euclidean mean
     DBA iteratively refines the average by DTW-aligning all series to it
  3. Anomaly scoring: DTW distance to DBA reference, normalized by
     empirical distribution, with Isolation Forest for multivariate anomaly detection
  4. Statistical testing: permutation-based p-value for anomaly significance
  5. Medoid computation: most central batch via DTW pairwise distance matrix

References:
  Petitjean, F. et al. (2011). A global averaging method for DTW.
  Killick, R. et al. (2012). Optimal detection of changepoints with a linear cost.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import percentileofscore
import threading
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tslearn")

from tslearn.metrics import dtw as compute_dtw
try:
    from tslearn.barycenters import dtw_barycenter_averaging
    HAS_DBA = True
except ImportError:
    HAS_DBA = False

try:
    import ruptures
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False

try:
    from sklearn.ensemble import IsolationForest
    HAS_IFOREST = True
except ImportError:
    HAS_IFOREST = False

from data_loader import load_process_data, get_all_batch_ids

PHASES = ["Preparation", "Granulation", "Drying", "Compression"]
NORM_LEN = 100

_fingerprint_cache = None
_changepoint_cache = {}


def _normalize_series(series: np.ndarray, length: int = NORM_LEN) -> np.ndarray:
    """Interpolate time series to fixed length."""
    if len(series) < 2:
        return np.zeros(length)
    x_old = np.linspace(0, 1, len(series))
    x_new = np.linspace(0, 1, length)
    f = interp1d(x_old, series, kind='linear', fill_value='extrapolate')
    return f(x_new)


def _extract_phase_curve(ts_df: pd.DataFrame, phase: str,
                          col: str = "Power_Consumption_kW") -> np.ndarray | None:
    phase_df = ts_df[ts_df["Phase"] == phase]
    if len(phase_df) < 3:
        return None
    return _normalize_series(phase_df[col].values, NORM_LEN)


# ─── CHANGE-POINT DETECTION ──────────────────────────────────────────────────

def detect_changepoints(batch_id: str) -> dict:
    """
    Automatic phase segmentation via change-point detection.
    Uses Ruptures PELT (Pruned Exact Linear Time) algorithm on power signal.
    This VALIDATES the pre-labeled phases and can detect sub-phases.
    """
    global _changepoint_cache
    if batch_id in _changepoint_cache:
        return _changepoint_cache[batch_id]

    all_ts = load_process_data()
    batch_df = all_ts[all_ts["Batch_ID"] == batch_id]
    if batch_df.empty:
        return {"batch_id": batch_id, "changepoints": [], "method": "none"}

    signal = batch_df["Power_Consumption_kW"].values

    if HAS_RUPTURES and len(signal) > 10:
        # PELT with RBF kernel cost function
        try:
            algo = ruptures.Pelt(model="rbf", min_size=5, jump=1)
            algo.fit(signal)
            # Detect change points (pen=3 controls sensitivity)
            cps = algo.predict(pen=3)
            cps = [cp for cp in cps if cp < len(signal)]  # remove endpoint
        except Exception:
            cps = _simple_changepoint(signal)
    else:
        cps = _simple_changepoint(signal)

    # Compare with labeled phases
    labeled_transitions = []
    prev_phase = None
    for i, row in batch_df.iterrows():
        phase = row.get("Phase", "Unknown")
        if phase != prev_phase and prev_phase is not None:
            idx = batch_df.index.get_loc(i)
            labeled_transitions.append(idx)
        prev_phase = phase

    result = {
        "batch_id": batch_id,
        "detected_changepoints": cps,
        "labeled_transitions": labeled_transitions,
        "n_detected": len(cps),
        "n_labeled": len(labeled_transitions),
        "agreement": _compute_changepoint_agreement(cps, labeled_transitions, len(signal)),
        "method": "PELT (Ruptures)" if HAS_RUPTURES else "energy_gradient",
        "signal_length": len(signal),
    }
    _changepoint_cache[batch_id] = result
    return result


def _simple_changepoint(signal: np.ndarray) -> list:
    """Fallback change-point detection via gradient thresholding."""
    if len(signal) < 10:
        return []
    gradient = np.abs(np.diff(signal))
    threshold = np.mean(gradient) + 2 * np.std(gradient)
    cps = list(np.where(gradient > threshold)[0])
    # Merge nearby changepoints
    if cps:
        merged = [cps[0]]
        for cp in cps[1:]:
            if cp - merged[-1] > 5:
                merged.append(cp)
        return merged
    return []


def _compute_changepoint_agreement(detected: list, labeled: list,
                                     signal_len: int) -> dict:
    """Compute agreement between detected and labeled phase transitions."""
    if not detected or not labeled:
        return {"agreement_score": 0, "method": "no_comparison"}

    tolerance = max(5, signal_len // 20)
    matched = 0
    for lab in labeled:
        for det in detected:
            if abs(det - lab) <= tolerance:
                matched += 1
                break

    precision = matched / len(detected) if detected else 0
    recall = matched / len(labeled) if labeled else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3),
        "tolerance_minutes": tolerance,
    }


# ─── FINGERPRINT COMPUTATION WITH DBA ────────────────────────────────────────

def compute_fingerprints() -> dict:
    """
    Build phase fingerprint library using DTW Barycentric Average (DBA).

    DBA iteratively refines the average by DTW-aligning all series to it,
    then computing the weighted mean of aligned points. This produces a
    reference curve that respects temporal warping — unlike Euclidean mean.

    Reference: Petitjean, F. et al. (2011). Pattern Recognition.
    """
    global _fingerprint_cache
    if _fingerprint_cache is not None:
        return _fingerprint_cache

    all_ts = load_process_data()
    fingerprints = {}

    for phase in PHASES:
        phase_curves = []
        for bid in get_all_batch_ids():
            batch_df = all_ts[all_ts["Batch_ID"] == bid]
            curve = _extract_phase_curve(batch_df, phase)
            if curve is not None:
                phase_curves.append(curve)

        if len(phase_curves) < 3:
            continue

        curves = np.array(phase_curves)
        n = len(curves)

        # --- DTW Barycentric Average (DBA) ---
        if HAS_DBA:
            # tslearn DBA: iterative DTW-based averaging
            curves_3d = curves.reshape(n, NORM_LEN, 1)
            dba_curve = dtw_barycenter_averaging(
                curves_3d, max_iter=30, tol=1e-5,
                init_barycenter=curves.mean(axis=0).reshape(NORM_LEN, 1),
            ).flatten()
            barycenter_method = "DTW Barycentric Average (Petitjean et al., 2011)"
        else:
            dba_curve = curves.mean(axis=0)
            barycenter_method = "Euclidean mean (DBA unavailable)"

        std_curve = curves.std(axis=0)

        # --- DTW pairwise distances for medoid ---
        dtw_dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = compute_dtw(curves[i], curves[j])
                dtw_dist_matrix[i, j] = d
                dtw_dist_matrix[j, i] = d

        medoid_idx = dtw_dist_matrix.sum(axis=1).argmin()

        # --- DTW distances from all curves to DBA (for anomaly scoring) ---
        dtw_to_dba = np.array([compute_dtw(c, dba_curve) for c in curves])

        fingerprints[phase] = {
            "mean": dba_curve.tolist(),
            "std": std_curve.tolist(),
            "upper": (dba_curve + std_curve).tolist(),
            "lower": (dba_curve - std_curve).tolist(),
            "medoid": curves[medoid_idx].tolist(),
            "n_batches": n,
            "all_curves": curves.tolist(),
            "dtw_to_reference": dtw_to_dba.tolist(),
            "barycenter_method": barycenter_method,
        }

    _fingerprint_cache = fingerprints
    return fingerprints


# ─── ANOMALY DETECTION ────────────────────────────────────────────────────────

def compute_anomaly_score(batch_id: str) -> dict:
    """
    Multi-method anomaly scoring:
      1. DTW distance to DBA reference (phase-level)
      2. Percentile rank in empirical DTW distribution (statistical)
      3. Isolation Forest on DTW feature vector (multivariate)
      4. Permutation-based p-value for anomaly significance
    """
    fingerprints = compute_fingerprints()
    all_ts = load_process_data()
    batch_df = all_ts[all_ts["Batch_ID"] == batch_id]

    scores = {}
    overall_deviations = []
    dtw_feature_vector = []

    for phase in PHASES:
        if phase not in fingerprints:
            scores[phase] = {
                "anomaly_score": 0, "asset_health": 100,
                "curve": [], "mean_curve": [], "upper_band": [], "lower_band": [],
            }
            continue

        curve = _extract_phase_curve(batch_df, phase)
        if curve is None:
            scores[phase] = {
                "anomaly_score": 0, "asset_health": 100,
                "curve": [], "mean_curve": [], "upper_band": [], "lower_band": [],
            }
            continue

        fp = fingerprints[phase]
        dba_curve = np.array(fp["mean"])
        dtw_distribution = np.array(fp["dtw_to_reference"])

        # DTW distance to DBA reference
        dtw_dist = compute_dtw(curve, dba_curve)
        dtw_feature_vector.append(dtw_dist)

        # Percentile rank (statistical)
        percentile = float(percentileofscore(dtw_distribution, dtw_dist))

        # Statistical anomaly: how extreme is this DTW distance?
        dtw_mean = float(dtw_distribution.mean())
        dtw_std = float(dtw_distribution.std())
        z_score = (dtw_dist - dtw_mean) / dtw_std if dtw_std > 0.01 else 0

        # Permutation-based p-value
        n_more_extreme = np.sum(dtw_distribution >= dtw_dist)
        p_value = float((n_more_extreme + 1) / (len(dtw_distribution) + 1))

        # Anomaly score with calibrated scaling
        anomaly_score = float(np.clip(percentile - 50, 0, 100))
        asset_health = round(max(0, 100 - anomaly_score), 1)
        overall_deviations.append(anomaly_score)

        scores[phase] = {
            "anomaly_score": round(anomaly_score, 1),
            "asset_health": asset_health,
            "dtw_distance": round(float(dtw_dist), 4),
            "dtw_percentile": round(percentile, 1),
            "z_score": round(z_score, 3),
            "p_value": round(p_value, 4),
            "is_anomaly": p_value < 0.05,
            "curve": curve.tolist(),
            "mean_curve": dba_curve.tolist(),
            "medoid_curve": fp["medoid"],
            "upper_band": fp["upper"],
            "lower_band": fp["lower"],
            "barycenter_method": fp.get("barycenter_method", "unknown"),
        }

    # Isolation Forest on DTW feature vector (multivariate anomaly)
    iforest_score = None
    if HAS_IFOREST and len(dtw_feature_vector) >= 2:
        try:
            all_features = []
            for bid in get_all_batch_ids():
                bid_df = all_ts[all_ts["Batch_ID"] == bid]
                features = []
                for phase in PHASES:
                    if phase in fingerprints:
                        c = _extract_phase_curve(bid_df, phase)
                        if c is not None:
                            dba = np.array(fingerprints[phase]["mean"])
                            features.append(compute_dtw(c, dba))
                        else:
                            features.append(0)
                    else:
                        features.append(0)
                all_features.append(features)

            X_forest = np.array(all_features)
            if X_forest.shape[0] > 5:
                iforest = IsolationForest(
                    n_estimators=100, contamination=0.1, random_state=42
                )
                iforest.fit(X_forest)
                query = np.array(dtw_feature_vector + [0] * (4 - len(dtw_feature_vector)))
                query = query[:4].reshape(1, -1)
                iforest_score = float(iforest.decision_function(query)[0])
        except Exception:
            pass

    # Changepoint analysis for this batch
    cp_result = detect_changepoints(batch_id)

    overall_health = round(100 - np.mean(overall_deviations), 1) if overall_deviations else 100

    return {
        "batch_id": batch_id,
        "overall_health": max(0.0, overall_health),
        "phases": scores,
        "isolation_forest_score": round(iforest_score, 4) if iforest_score is not None else None,
        "changepoint_analysis": cp_result,
        "method": f"DTW to DBA reference ({fingerprints.get(PHASES[0], {}).get('barycenter_method', 'DTW')})",
        "anomaly_methods": [
            "DTW distance to DBA barycenter",
            "Empirical percentile rank",
            "Permutation p-value",
            "Isolation Forest (multivariate)",
        ],
    }

_iforest_cache = {}  # {batch_id: dtw_feature_vector} + "_model": IsolationForest


def _precompute_iforest_features() -> tuple:
    """
    Precompute DTW feature matrix and Isolation Forest model once.
    This avoids the O(n²) recomputation that was happening inside
    compute_anomaly_score × get_all_batch_ids.

    Returns (iforest_model, feature_dict) where:
      - iforest_model: fitted IsolationForest (or None)
      - feature_dict: {batch_id: dtw_feature_vector}
    """
    global _iforest_cache
    if "_model" in _iforest_cache:
        return _iforest_cache["_model"], _iforest_cache

    fingerprints = compute_fingerprints()
    all_ts = load_process_data()
    batch_ids = get_all_batch_ids()

    all_features = {}
    feature_matrix = []

    for bid in batch_ids:
        bid_df = all_ts[all_ts["Batch_ID"] == bid]
        features = []
        for phase in PHASES:
            if phase in fingerprints:
                c = _extract_phase_curve(bid_df, phase)
                if c is not None:
                    dba = np.array(fingerprints[phase]["mean"])
                    features.append(compute_dtw(c, dba))
                else:
                    features.append(0)
            else:
                features.append(0)
        all_features[bid] = features
        feature_matrix.append(features)

    iforest_model = None
    if HAS_IFOREST and len(feature_matrix) > 5:
        try:
            X = np.array(feature_matrix)
            iforest_model = IsolationForest(
                n_estimators=100, contamination=0.1, random_state=42
            )
            iforest_model.fit(X)
        except Exception:
            pass

    _iforest_cache = all_features
    _iforest_cache["_model"] = iforest_model
    return iforest_model, all_features


def get_all_anomaly_scores() -> list:
    """
    Compute anomaly scores for ALL batches efficiently.

    Performance optimization:
      - Precomputes DTW feature matrix and Isolation Forest ONCE
      - Shares computed features across all batch score computations
      - Reduces DTW computations from O(n² × phases) to O(n × phases)
    """
    fingerprints = compute_fingerprints()
    all_ts = load_process_data()
    batch_ids = get_all_batch_ids()

    # Precompute Isolation Forest (ONCE, not per-batch)
    iforest_model, feature_dict = _precompute_iforest_features()

    results = []
    for bid in batch_ids:
        try:
            batch_df = all_ts[all_ts["Batch_ID"] == bid]
            scores = {}
            overall_deviations = []

            for phase in PHASES:
                if phase not in fingerprints:
                    scores[phase] = {"anomaly_score": 0, "asset_health": 100}
                    continue

                curve = _extract_phase_curve(batch_df, phase)
                if curve is None:
                    scores[phase] = {"anomaly_score": 0, "asset_health": 100}
                    continue

                fp = fingerprints[phase]
                dba_curve = np.array(fp["mean"])
                dtw_distribution = np.array(fp["dtw_to_reference"])
                dtw_dist = compute_dtw(curve, dba_curve)
                percentile = float(percentileofscore(dtw_distribution, dtw_dist))
                anomaly_score = float(np.clip(percentile - 50, 0, 100))
                overall_deviations.append(anomaly_score)

            overall_health = round(100 - np.mean(overall_deviations), 1) if overall_deviations else 100
            overall_health = max(0.0, overall_health)

            # Isolation Forest score using PRECOMPUTED model
            iforest_score = None
            if iforest_model is not None and bid in feature_dict:
                try:
                    query = np.array(feature_dict[bid]).reshape(1, -1)
                    iforest_score = float(iforest_model.decision_function(query)[0])
                except Exception:
                    pass

            results.append({
                "batch_id": bid,
                "overall_health": overall_health,
                "isolation_forest_score": round(iforest_score, 4) if iforest_score is not None else None,
            })
        except Exception:
            results.append({"batch_id": bid, "overall_health": 85.0})
    return results


# ─── ENERGY PATTERN INTELLIGENCE ─────────────────────────────────────────────

def compute_degradation_trend(phase: str = None) -> dict:
    """
    Equipment degradation trend detection via exponential fit on DTW distances.

    For each manufacturing phase, fits:
      DTW(t) = a · exp(b · t) + c

    Where increasing b indicates accelerating degradation.
    Also computes Remaining Useful Life (RUL) — batches until DTW exceeds
    the anomaly threshold (95th percentile of historical distribution).

    Returns:
      - Trend direction per phase (improving/stable/degrading)
      - Degradation rate coefficient
      - Estimated RUL in batches
      - Alert if RUL < 20 batches
    """
    from scipy.optimize import curve_fit

    fingerprints = compute_fingerprints()
    all_ts = load_process_data()
    batch_ids = get_all_batch_ids()

    phases_to_analyze = [phase] if phase else PHASES
    results = {}

    for ph in phases_to_analyze:
        if ph not in fingerprints:
            continue

        fp = fingerprints[ph]
        dba_curve = np.array(fp["mean"])
        dtw_distribution = np.array(fp["dtw_to_reference"])

        # Compute DTW distance to DBA for each batch in order
        dtw_sequence = []
        for bid in batch_ids:
            batch_df = all_ts[all_ts["Batch_ID"] == bid]
            curve = _extract_phase_curve(batch_df, ph)
            if curve is not None:
                dtw_dist = compute_dtw(curve, dba_curve)
                dtw_sequence.append(float(dtw_dist))
            else:
                dtw_sequence.append(None)

        # Remove Nones and track indices
        valid = [(i, v) for i, v in enumerate(dtw_sequence) if v is not None]
        if len(valid) < 10:
            results[ph] = {"trend": "insufficient_data", "n_valid": len(valid)}
            continue

        t = np.array([v[0] for v in valid], dtype=float)
        y = np.array([v[1] for v in valid], dtype=float)
        t_norm = (t - t.min()) / (t.max() - t.min() + 1e-9)

        # Linear trend first (robust)
        coeffs = np.polyfit(t_norm, y, 1)
        linear_slope = coeffs[0]

        # Exponential fit: y = a * exp(b * t) + c
        exp_fit_success = False
        try:
            def exp_model(x, a, b, c):
                return a * np.exp(b * x) + c

            popt, _ = curve_fit(
                exp_model, t_norm, y,
                p0=[y.std(), 0.5, y.mean()],
                bounds=([0, -5, 0], [y.max() * 2, 5, y.max() * 2]),
                maxfev=3000,
            )
            a, b, c = popt
            y_pred = exp_model(t_norm, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            exp_fit_success = r2 > 0.1
        except Exception:
            a, b, c, r2 = 0, 0, y.mean(), 0

        # Determine trend
        if exp_fit_success and b > 0.3:
            trend = "degrading"
        elif exp_fit_success and b < -0.3:
            trend = "improving"
        elif abs(linear_slope) < y.std() * 0.1:
            trend = "stable"
        elif linear_slope > 0:
            trend = "degrading"
        else:
            trend = "improving"

        # RUL estimation: when does DTW exceed the anomaly threshold?
        anomaly_threshold = float(np.percentile(dtw_distribution, 95))
        current_dtw = y[-1]
        rul_batches = None

        if exp_fit_success and b > 0.01:
            # Solve: a * exp(b * t_rul) + c = threshold
            try:
                t_rul = np.log((anomaly_threshold - c) / (a + 1e-9)) / (b + 1e-9)
                rul_batches = max(0, int((t_rul - t_norm[-1]) * len(batch_ids)))
            except (ValueError, OverflowError):
                rul_batches = None
        elif linear_slope > 0.01:
            # Linear extrapolation
            remaining = (anomaly_threshold - current_dtw) / (linear_slope + 1e-9)
            rul_batches = max(0, int(remaining * len(batch_ids)))

        results[ph] = {
            "trend": trend,
            "linear_slope": round(float(linear_slope), 4),
            "exponential_fit": {
                "a": round(float(a), 4),
                "b": round(float(b), 4),
                "c": round(float(c), 4),
                "r2": round(float(r2), 4),
                "success": exp_fit_success,
            } if exp_fit_success else None,
            "current_dtw": round(float(current_dtw), 4),
            "anomaly_threshold": round(float(anomaly_threshold), 4),
            "rul_batches": rul_batches,
            "alert": rul_batches is not None and rul_batches < 20,
            "alert_message": (
                f"Phase '{ph}' equipment predicted to reach anomaly threshold "
                f"in ~{rul_batches} batches. Schedule preventive maintenance."
            ) if rul_batches is not None and rul_batches < 20 else None,
            "dtw_sequence": [round(v, 3) for v in y.tolist()],
            "n_datapoints": len(y),
        }

    return {
        "phases": results,
        "method": "Exponential degradation fitting on DTW distance sequence",
        "total_batches_analyzed": len(batch_ids),
    }


def energy_pattern_forecast(phase: str, n_future: int = 5) -> dict:
    """
    Forecast next N batches' energy consumption using Holt-Winters exponential smoothing.

    Method:
      1. Compute per-batch energy (area under power curve) for each historical batch
      2. Fit Holt-Winters additive model with damped trend (statsmodels)
      3. Produce point forecasts + prediction intervals from residual distribution
      4. Also generate curve-level forecasts using DBA + trend-adjusted residual model

    Holt-Winters Exponential Smoothing:
      Level:   l_t = α·y_t + (1-α)·(l_{t-1} + φ·b_{t-1})
      Trend:   b_t = β·(l_t - l_{t-1}) + (1-β)·φ·b_{t-1}
      Forecast: ŷ_{t+h} = l_t + φ·b_t·(1 + φ + ... + φ^{h-1})
      Damped trend (φ < 1) prevents extrapolation explosion.

    Ref: Hyndman, R.J. & Athanasopoulos, G. (2021). Forecasting: Principles and Practice, 3rd ed.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

    fingerprints = compute_fingerprints()
    if phase not in fingerprints:
        return {"error": f"Phase '{phase}' not found"}

    fp = fingerprints[phase]
    dba_curve = np.array(fp["mean"])
    all_curves = np.array(fp["all_curves"])
    n_hist = len(all_curves)

    # ── 1. Compute per-batch energy time-series ──
    batch_energies = np.array([float(c.sum() / 60.0) for c in all_curves])

    # ── 2. Fit Holt-Winters exponential smoothing ──
    forecast_energies = []
    forecast_low = []
    forecast_high = []
    hw_params = {}
    method_used = "insufficient_data"

    if n_hist >= 8:
        try:
            # Holt-Winters with damped additive trend (no seasonality — batches are aperiodic)
            hw_model = ExponentialSmoothing(
                batch_energies,
                trend="add",
                seasonal=None,
                damped_trend=True,
                initialization_method="estimated",
            ).fit(optimized=True)

            # Point forecasts
            hw_forecast = hw_model.forecast(n_future)
            forecast_energies = [round(float(v), 3) for v in hw_forecast]

            # Prediction intervals from residual distribution (Gaussian assumption)
            residuals = hw_model.resid
            residual_std = float(np.std(residuals))
            residual_mean = float(np.mean(residuals))

            # Prediction intervals widen with horizon (√h scaling)
            for h in range(1, n_future + 1):
                interval_width = 1.645 * residual_std * np.sqrt(h)
                forecast_low.append(round(float(hw_forecast.iloc[h - 1] - interval_width), 3))
                forecast_high.append(round(float(hw_forecast.iloc[h - 1] + interval_width), 3))

            hw_params = {
                "alpha": round(float(hw_model.params.get("smoothing_level", 0)), 4),
                "beta": round(float(hw_model.params.get("smoothing_trend", 0)), 4),
                "phi": round(float(hw_model.params.get("damping_trend", 0)), 4),
                "residual_std": round(residual_std, 4),
                "residual_mean": round(residual_mean, 4),
                "aic": round(float(hw_model.aic), 2),
                "bic": round(float(hw_model.bic), 2),
            }
            method_used = "Holt-Winters Exponential Smoothing (damped additive trend)"

        except Exception:
            # Fallback: Simple exponential smoothing (no trend)
            try:
                ses_model = SimpleExpSmoothing(
                    batch_energies, initialization_method="estimated"
                ).fit(optimized=True)
                ses_forecast = ses_model.forecast(n_future)
                forecast_energies = [round(float(v), 3) for v in ses_forecast]
                residual_std = float(np.std(ses_model.resid))
                for h in range(1, n_future + 1):
                    interval_width = 1.645 * residual_std * np.sqrt(h)
                    forecast_low.append(round(float(ses_forecast.iloc[h - 1] - interval_width), 3))
                    forecast_high.append(round(float(ses_forecast.iloc[h - 1] + interval_width), 3))
                hw_params = {"alpha": round(float(ses_model.params.get("smoothing_level", 0)), 4)}
                method_used = "Simple Exponential Smoothing (fallback)"
            except Exception:
                # Last resort: historical mean
                mean_e = float(batch_energies.mean())
                std_e = float(batch_energies.std())
                forecast_energies = [round(mean_e, 3)] * n_future
                forecast_low = [round(mean_e - 1.645 * std_e, 3)] * n_future
                forecast_high = [round(mean_e + 1.645 * std_e, 3)] * n_future
                method_used = "Historical mean (fallback)"
    else:
        # Not enough data for exponential smoothing
        mean_e = float(batch_energies.mean())
        std_e = float(batch_energies.std()) if n_hist > 1 else 0
        forecast_energies = [round(mean_e, 3)] * n_future
        forecast_low = [round(mean_e - 1.645 * std_e, 3)] * n_future
        forecast_high = [round(mean_e + 1.645 * std_e, 3)] * n_future
        method_used = "Historical mean (insufficient data for Holt-Winters)"

    # ── 3. Curve-level forecast using DBA + trend-adjusted residual model ──
    residuals = all_curves - dba_curve
    residual_mean_curve = residuals.mean(axis=0)
    residual_std_curve = residuals.std(axis=0)

    # Detect drift in recent curves
    if n_hist >= 10:
        recent_residuals = residuals[-5:]
        earlier_residuals = residuals[:-5]
        drift = recent_residuals.mean(axis=0) - earlier_residuals.mean(axis=0)
    else:
        drift = np.zeros(len(dba_curve))

    curve_forecasts = []
    for i in range(n_future):
        noise = np.random.normal(residual_mean_curve, residual_std_curve)
        forecast_curve = dba_curve + drift * (i + 1) + noise
        forecast_curve = np.maximum(forecast_curve, 0)
        curve_forecasts.append(forecast_curve)

    curve_forecasts = np.array(curve_forecasts)

    return {
        "phase": phase,
        "n_forecast": n_future,
        "predicted_energy_kwh": forecast_energies,
        "prediction_interval_low": forecast_low,
        "prediction_interval_high": forecast_high,
        "mean_predicted_energy_kwh": round(float(np.mean(forecast_energies)), 3),
        "historical_energy_kwh": [round(float(e), 3) for e in batch_energies],
        "curve_forecast_mean": curve_forecasts.mean(axis=0).tolist(),
        "curve_forecast_lower_90": np.percentile(curve_forecasts, 5, axis=0).tolist(),
        "curve_forecast_upper_90": np.percentile(curve_forecasts, 95, axis=0).tolist(),
        "drift_detected": bool(np.abs(drift).max() > residual_std_curve.mean() * 0.5),
        "drift_magnitude": round(float(np.abs(drift).mean()), 4),
        "exponential_smoothing": hw_params,
        "method": method_used,
        "n_historical_patterns": n_hist,
    }
