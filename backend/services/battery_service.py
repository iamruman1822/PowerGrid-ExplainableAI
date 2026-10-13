"""
Battery Service
===============
Reads battery degradation outputs (summaries + per-cycle predictions)
from the ZerodayExplorers_BatteryDegradation module.
"""

import os
import pandas as pd
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BATTERY_DIR = os.path.join(ROOT_DIR, "ZerodayExplorers_BatteryDegradation", "outputs")
SUMMARY_CSV = os.path.join(BATTERY_DIR, "data", "battery_full_summary.csv")
FULL_PREDS_CSV = os.path.join(BATTERY_DIR, "data", "nasa_2yr_all_batteries.csv")
NASA_2YR_CSV = os.path.join(BATTERY_DIR, "data", "nasa_2yr_all_batteries.csv")
PLOT_DIR = os.path.join(BATTERY_DIR, "plots")

_summary_cache = None
_predictions_cache = None


def _load_summary_from_db():
    try:
        from database import get_db_connection
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT battery, total_cycles,
                           initial_capacity_ahr AS "initial_capacity_Ahr",
                           final_actual_cap_ahr AS "final_actual_cap_Ahr",
                           final_pred_cap_ahr AS "final_pred_cap_Ahr",
                           soh_actual_pct AS "soh_actual_%",
                           soh_predicted_pct AS "soh_predicted_%",
                           rul_predicted_cycles,
                           rul_actual_cycles,
                           avg_deg_rate_ahr_cycle AS "avg_deg_rate_Ahr_cycle",
                           anomaly_count,
                           mae_ahr AS "mae_Ahr",
                           rmse_ahr AS "rmse_Ahr"
                    FROM battery_fleet
                    ORDER BY battery ASC;
                """)
                rows = cur.fetchall()
                if rows:
                    return pd.DataFrame(rows)
        finally:
            conn.close()
    except Exception as e:
        print(f"[WARN] Failed to load battery fleet from PostgreSQL: {e}")
    return None


def _load_summary():
    global _summary_cache
    if _summary_cache is not None:
        return _summary_cache

    # 1. Try PostgreSQL
    db_df = _load_summary_from_db()
    if db_df is not None and not db_df.empty:
        _summary_cache = db_df
        return db_df

    # 2. Fallback to CSV
    if not os.path.exists(SUMMARY_CSV):
        return pd.DataFrame()
    _summary_cache = pd.read_csv(SUMMARY_CSV)
    return _summary_cache


def _load_predictions():
    global _predictions_cache
    if _predictions_cache is not None:
        return _predictions_cache
    if not os.path.exists(FULL_PREDS_CSV):
        return pd.DataFrame()
    _predictions_cache = pd.read_csv(FULL_PREDS_CSV)
    return _predictions_cache


def get_battery_summary():
    """Return per-battery health summary."""
    df = _load_summary()
    if df.empty:
        return []
    # Replace NaN with None for JSON
    return df.where(df.notna(), None).to_dict(orient="records")


def get_fleet_overview():
    """Return aggregated fleet health metrics for dashboard."""
    df = _load_summary()
    if df.empty:
        return {}

    soh_col = "soh_actual_%" if "soh_actual_%" in df.columns else "soh_predicted_%"
    soh_vals = pd.to_numeric(df[soh_col], errors="coerce").dropna()

    # Cap SoH at 100% — some NASA batteries (B0038/B0039/B0041) have corrupted
    # initial capacity measurements producing physically impossible values > 100%.
    # SoH cannot exceed 100% by definition; clamp before aggregating.
    soh_vals_capped = soh_vals.clip(upper=100.0)

    return {
        "total_batteries": len(df),
        "avg_soh_pct": round(float(soh_vals_capped.mean()), 1) if len(soh_vals_capped) > 0 else 0,
        "min_soh_pct": round(float(soh_vals_capped.min()), 1) if len(soh_vals_capped) > 0 else 0,
        "max_soh_pct": round(float(soh_vals_capped.max()), 1) if len(soh_vals_capped) > 0 else 0,
        "total_anomalies": int(df["anomaly_count"].sum()) if "anomaly_count" in df.columns else 0,
        "batteries_at_risk": int((soh_vals_capped < 80).sum()) if len(soh_vals_capped) > 0 else 0,
    }


def get_battery_cycles(battery_id=None):
    """Return per-cycle prediction data, optionally filtered by battery ID."""
    df = _load_predictions()
    if df.empty:
        return []
    if battery_id:
        df = df[df["battery"] == battery_id]
    # Limit to prevent huge payloads
    if len(df) > 2000:
        df = df.tail(2000)
    return df.where(df.notna(), None).to_dict(orient="records")


def get_battery_ids():
    """Return list of unique battery IDs."""
    df = _load_summary()
    if df.empty:
        return []
    return df["battery"].tolist()


def get_plot_path(filename):
    """Return full path to a battery plot image if it exists."""
    path = os.path.join(PLOT_DIR, filename)
    if os.path.exists(path):
        return path
    return None
