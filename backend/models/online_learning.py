"""
Online Learning Manager — Continuous Surrogate & Model Retraining

Implements true online learning for the manufacturing intelligence platform:
  1. Accumulates new batch data as it arrives
  2. Triggers model retraining when sufficient new data is collected
  3. Retrains LightGBM surrogates, SCM structural equations, and physics calibration
  4. Tracks model versions for reproducibility
  5. Validates new models against holdout data before promoting

Architecture:
  - IngestBatch → accumulate in buffer
  - When buffer reaches threshold → retrain all models
  - Compare new model R² vs old model R²
  - If new ≥ old × (1 - tolerance): promote new model
  - Otherwise: keep old model, log warning

This addresses the "continuous learning" requirement: the optimization
landscape itself evolves, not just the Bayesian parameter estimates.
"""

import numpy as np
import pandas as pd
import logging
import threading
from typing import Dict, Optional, Any
from datetime import datetime
from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)


class ModelVersion:
    """Tracks a single model version."""
    def __init__(self, version: int, model_type: str, r2: float, n_training: int):
        self.version = version
        self.model_type = model_type
        self.r2 = r2
        self.n_training = n_training
        self.created_at = datetime.now().isoformat()
        self.is_active = True

    def to_dict(self):
        return {
            "version": self.version,
            "model_type": self.model_type,
            "r2": round(self.r2, 4),
            "n_training_samples": self.n_training,
            "created_at": self.created_at,
            "is_active": self.is_active,
        }


class OnlineLearningManager:
    """
    Manages online retraining of all surrogate models.

    Tracks:
      - LightGBM surrogates (per-target)
      - SCM structural equations
      - Physics engine calibration

    Retraining is triggered when:
      - N new batches have been ingested (default: 10)
      - Or manual retrain is requested via API

    Model promotion uses a validation check:
      - New model R² must be within tolerance of old model R²
      - Prevents catastrophic model degradation from noisy data
    """

    def __init__(self, retrain_threshold: int = 10, r2_tolerance: float = 0.05):
        """
        Args:
            retrain_threshold: retrain after this many new batches
            r2_tolerance: new model must achieve r2 >= old_r2 * (1 - tolerance)
        """
        self.retrain_threshold = retrain_threshold
        self.r2_tolerance = r2_tolerance
        self.new_batch_buffer = []
        self.model_versions: Dict[str, list] = {}
        self.current_version = 0
        self.retrain_history = []
        self._last_retrain_n = 0
        self._lock = threading.RLock()

    def ingest_batch(self, batch_data: dict) -> dict:
        """
        Ingest a new batch's data into the buffer.

        Args:
            batch_data: dict containing all input params + output targets

        Returns:
            dict with buffer status and whether retraining was triggered
        """
        with self._lock:
            self.new_batch_buffer.append({
                **batch_data,
                "ingested_at": datetime.now().isoformat(),
            })

            result = {
                "ingested": True,
                "buffer_size": len(self.new_batch_buffer),
                "retrain_at": self.retrain_threshold,
                "retrained": False,
            }

            if len(self.new_batch_buffer) >= self.retrain_threshold:
                retrain_result = self.retrain_all()
                result["retrained"] = True
                result["retrain_result"] = retrain_result

            return result

    def retrain_all(self) -> dict:
        """
        Retrain all models with original + new data.

        Pipeline:
          1. Combine original training data with new batch buffer
          2. Retrain LightGBM surrogates
          3. Refit SCM structural equations
          4. Recalibrate physics engine
          5. Validate against holdout
          6. Promote or reject each model
        """
        from models.causal import (
            _get_enriched_data, _lgb_models, _scaler, _param_bounds,
            INPUT_PARAMS, OUTPUT_TARGETS, _scm, _fit_scm,
        )
        from models.physics import get_physics_engine
        from sklearn.preprocessing import StandardScaler
        import lightgbm as lgb

        df = _get_enriched_data()

        # Add buffered data to training set
        if self.new_batch_buffer:
            new_df = pd.DataFrame(self.new_batch_buffer)
            # Only add columns that exist in original data
            common_cols = [c for c in new_df.columns
                          if c in df.columns and c != "ingested_at"]
            if common_cols:
                new_rows = new_df[common_cols]
                df = pd.concat([df, new_rows], ignore_index=True)
                logger.info(f"Extended training data: {len(df)} total rows "
                           f"(+{len(new_rows)} new)")

        self.current_version += 1
        retrain_results = {
            "version": self.current_version,
            "n_new_batches": len(self.new_batch_buffer),
            "total_training_samples": len(df),
            "models": {},
        }

        # Holdout split for validation (last 15% of data)
        n = len(df)
        split_idx = int(n * 0.85)
        df_train = df.iloc[:split_idx]
        df_val = df.iloc[split_idx:]

        X_train = df_train[INPUT_PARAMS].values
        X_val = df_val[INPUT_PARAMS].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        # --- Retrain LightGBM surrogates ---
        all_targets = OUTPUT_TARGETS + ["Power_kWh", "Quality_Score", "Yield_Score"]
        lgb_results = {}

        for target in all_targets:
            if target not in df.columns:
                continue

            y_train = df_train[target].values
            y_val = df_val[target].values

            # Train new model
            new_model = lgb.LGBMRegressor(
                n_estimators=200, learning_rate=0.05,
                num_leaves=15, random_state=42, verbose=-1,
            )
            new_model.fit(X_train_s, y_train)

            # Validate
            new_r2 = r2_score(y_val, new_model.predict(X_val_s))

            # Check old model R²
            old_r2 = -1.0
            if target in _lgb_models:
                try:
                    old_pred = _lgb_models[target].predict(X_val_s)
                    old_r2 = r2_score(y_val, old_pred)
                except Exception:
                    old_r2 = -1.0

            # Promotion check
            promoted = new_r2 >= old_r2 * (1 - self.r2_tolerance) or old_r2 < 0
            if promoted:
                # Update global model
                _lgb_models[target] = new_model
                logger.info(f"Model promoted: {target} R²={new_r2:.4f} "
                           f"(was {old_r2:.4f})")
            else:
                logger.warning(f"Model rejected: {target} new R²={new_r2:.4f} "
                              f"< old R²={old_r2:.4f}")

            lgb_results[target] = {
                "old_r2": round(old_r2, 4),
                "new_r2": round(new_r2, 4),
                "promoted": promoted,
            }

            # Track version
            mv = ModelVersion(
                self.current_version, f"lgbm_{target}",
                new_r2, len(df_train)
            )
            mv.is_active = promoted
            if target not in self.model_versions:
                self.model_versions[target] = []
            self.model_versions[target].append(mv)

        retrain_results["models"]["lgbm"] = lgb_results

        # --- Update scaler globally ---
        import models.causal as causal_module
        causal_module._scaler = scaler

        # --- Refit SCM ---
        try:
            scm_results = _fit_scm()
            retrain_results["models"]["scm"] = {
                "refitted": True,
                "n_equations": len(scm_results),
            }
            logger.info(f"SCM refitted: {len(scm_results)} equations")
        except Exception as e:
            retrain_results["models"]["scm"] = {
                "refitted": False, "error": str(e)
            }

        # --- Recalibrate physics ---
        try:
            physics = get_physics_engine()
            cal = physics.calibrate(df)
            retrain_results["models"]["physics"] = {
                "recalibrated": True,
                "r2_scores": cal,
            }
            logger.info(f"Physics recalibrated: {cal}")
        except Exception as e:
            retrain_results["models"]["physics"] = {
                "recalibrated": False, "error": str(e)
            }

        # Clear buffer
        self._last_retrain_n = len(self.new_batch_buffer)
        self.new_batch_buffer = []

        # Record history
        self.retrain_history.append({
            "version": self.current_version,
            "timestamp": datetime.now().isoformat(),
            "n_new_batches": self._last_retrain_n,
            "total_samples": len(df),
            "lgbm_results": lgb_results,
        })

        return retrain_results

    def get_state(self) -> dict:
        """Full state for API."""
        return {
            "current_version": self.current_version,
            "buffer_size": len(self.new_batch_buffer),
            "retrain_threshold": self.retrain_threshold,
            "r2_tolerance": self.r2_tolerance,
            "total_retrains": len(self.retrain_history),
            "model_versions": {
                target: [mv.to_dict() for mv in versions[-3:]]
                for target, versions in self.model_versions.items()
            },
            "last_retrain": self.retrain_history[-1] if self.retrain_history else None,
        }


# Module-level singleton
_online_manager = None


def get_online_manager() -> OnlineLearningManager:
    global _online_manager
    if _online_manager is None:
        _online_manager = OnlineLearningManager(retrain_threshold=10)
    return _online_manager
