import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
PROCESS_FILE = DATA_DIR / "_h_batch_process_data.xlsx"
PRODUCTION_FILE = DATA_DIR / "_h_batch_production_data.xlsx"

INDIA_EMISSION_FACTOR = 0.82  # kg CO2e per kWh

_process_cache = None
_production_cache = None
_summary_cache = None


def load_production_data() -> pd.DataFrame:
    global _production_cache
    if _production_cache is None:
        _production_cache = pd.read_excel(PRODUCTION_FILE, sheet_name="BatchData")
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

        # Quality score composite: higher dissolution + hardness, lower friability + disintegration time
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
