import pandas as pd
import numpy as np
import logging
from pathlib import Path
from pydantic import BaseModel, validator
from typing import Optional, List

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
PROCESS_FILE = DATA_DIR / "_h_batch_process_data.xlsx"
PRODUCTION_FILE = DATA_DIR / "_h_batch_production_data.xlsx"

INDIA_EMISSION_FACTOR = 0.82  # kg CO2e per kWh (India grid, CEA 2023)

_process_cache = None
_production_cache = None
_summary_cache = None
_data_quality_report = None


# ─── VALIDATION SCHEMAS ──────────────────────────────────────────────────────

class BatchProductionSchema(BaseModel):
    """Pydantic validation schema for batch production data."""
    Batch_ID: str
    Granulation_Time: float
    Binder_Amount: float
    Drying_Temp: float
    Drying_Time: float
    Compression_Force: float
    Machine_Speed: float
    Lubricant_Conc: float
    Hardness: float
    Friability: float
    Disintegration_Time: float
    Dissolution_Rate: float
    Content_Uniformity: float

    @validator("Hardness")
    def hardness_positive(cls, v):
        if v < 0:
            raise ValueError("Hardness cannot be negative")
        return v

    @validator("Friability")
    def friability_bounded(cls, v):
        if v < 0 or v > 10:
            raise ValueError("Friability must be in [0, 10]")
        return v

    @validator("Dissolution_Rate")
    def dissolution_bounded(cls, v):
        return max(0, min(v, 100))


class ProcessDataSchema(BaseModel):
    """Pydantic validation for time-series process data."""
    Batch_ID: str
    Time_Minutes: float
    Temperature_C: float
    Pressure_Bar: float
    Power_Consumption_kW: float
    Vibration_mm_s: float
    Phase: str


# ─── DATA QUALITY ────────────────────────────────────────────────────────────

def _detect_outliers_iqr(series: pd.Series, factor: float = 3.0) -> pd.Series:
    """IQR-based outlier detection. Returns boolean mask of outliers."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    return (series < lower) | (series > upper)


def _impute_and_clean(df: pd.DataFrame, numeric_cols: list) -> tuple:
    """
    Impute missing values + detect outliers.
    Strategy: median imputation for missing, IQR flagging for outliers.
    Returns (cleaned_df, quality_report).
    """
    report = {"missing": {}, "outliers": {}, "imputed_count": 0}

    for col in numeric_cols:
        if col not in df.columns:
            continue

        # Count missing
        n_missing = int(df[col].isna().sum())
        if n_missing > 0:
            report["missing"][col] = n_missing
            # Median imputation
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            report["imputed_count"] += n_missing
            logger.info(f"Imputed {n_missing} missing values in '{col}' with median={median_val:.3f}")

        # Outlier detection
        outlier_mask = _detect_outliers_iqr(df[col])
        n_outliers = int(outlier_mask.sum())
        if n_outliers > 0:
            report["outliers"][col] = {
                "count": n_outliers,
                "indices": df[outlier_mask].index.tolist()[:10],  # first 10
            }

    report["total_rows"] = len(df)
    report["data_health"] = "clean" if not report["missing"] and not report["outliers"] else "has_issues"
    return df, report


# ─── DATA LOADING ────────────────────────────────────────────────────────────

def load_production_data() -> pd.DataFrame:
    global _production_cache, _data_quality_report
    if _production_cache is None:
        df = pd.read_excel(PRODUCTION_FILE, sheet_name="BatchData")

        # Validate and clean
        numeric_cols = [
            "Granulation_Time", "Binder_Amount", "Drying_Temp", "Drying_Time",
            "Compression_Force", "Machine_Speed", "Lubricant_Conc",
            "Hardness", "Friability", "Disintegration_Time",
            "Dissolution_Rate", "Content_Uniformity",
        ]
        df, quality_report = _impute_and_clean(df, numeric_cols)
        _data_quality_report = quality_report
        logger.info(f"Production data loaded: {len(df)} rows, quality={quality_report['data_health']}")
        _production_cache = df
    return _production_cache.copy()


def load_process_data(batch_id: str = None) -> pd.DataFrame:
    """Load time-series data for one or all batches."""
    global _process_cache
    if _process_cache is None:
        xl = pd.ExcelFile(PROCESS_FILE)
        batch_sheets = [s for s in xl.sheet_names if s.startswith("Batch_")]
        dfs = []
        for sheet in batch_sheets:
            df = xl.parse(sheet)
            dfs.append(df)
        _process_cache = pd.concat(dfs, ignore_index=True)

        # Clean process data
        numeric_cols = ["Temperature_C", "Pressure_Bar", "Power_Consumption_kW", "Vibration_mm_s"]
        _process_cache, _ = _impute_and_clean(_process_cache, numeric_cols)
        logger.info(f"Process data loaded: {len(_process_cache)} rows")

    df = _process_cache.copy()
    if batch_id:
        df = df[df["Batch_ID"] == batch_id]
    return df


def load_summary_data() -> pd.DataFrame:
    global _summary_cache
    if _summary_cache is None:
        xl = pd.ExcelFile(PROCESS_FILE)
        _summary_cache = xl.parse("Summary")
    return _summary_cache.copy()


def get_all_batch_ids() -> list:
    prod = load_production_data()
    return sorted(prod["Batch_ID"].tolist())


def get_data_quality_report() -> dict:
    """Return data quality report from last load."""
    if _data_quality_report is None:
        load_production_data()  # trigger load
    return _data_quality_report or {}


def get_batch_summary(batch_id: str) -> dict:
    """Merge process summary + production quality for a batch."""
    prod = load_production_data()
    prod_row = prod[prod["Batch_ID"] == batch_id]
    if prod_row.empty:
        return {}
    prod_dict = prod_row.iloc[0].to_dict()

    # Compute energy from time-series
    ts = load_process_data(batch_id)
    if not ts.empty:
        duration_h = len(ts) / 60.0
        avg_power = ts["Power_Consumption_kW"].mean()
        total_kwh = avg_power * duration_h
        carbon_kg = total_kwh * INDIA_EMISSION_FACTOR
        phases = ts["Phase"].unique().tolist()
        prod_dict["total_kwh"] = round(total_kwh, 2)
        prod_dict["avg_power_kw"] = round(avg_power, 2)
        prod_dict["carbon_kg_co2e"] = round(carbon_kg, 2)
        prod_dict["duration_minutes"] = len(ts)
        prod_dict["phases"] = phases
        prod_dict["max_temperature"] = round(ts["Temperature_C"].max(), 2)
        prod_dict["avg_vibration"] = round(ts["Vibration_mm_s"].mean(), 4)

        # Quality score composite
        h = prod_dict.get("Hardness", 80)
        d = prod_dict.get("Dissolution_Rate", 85)
        f = prod_dict.get("Friability", 1.0)
        di = prod_dict.get("Disintegration_Time", 10)
        cu = prod_dict.get("Content_Uniformity", 98)
        quality_score = (
            (min(h, 120) / 120) * 30
            + (min(d, 100) / 100) * 30
            + (1 - min(f, 2) / 2) * 15
            + (1 - min(di, 20) / 20) * 10
            + (min(cu, 105) / 105) * 15
        )
        prod_dict["quality_score"] = round(quality_score, 1)

    return prod_dict

