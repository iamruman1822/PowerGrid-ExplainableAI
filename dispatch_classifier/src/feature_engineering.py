"""
Feature Engineering Module
Constructs 34 microgrid operational indicators from forecast and battery data.
"""

import numpy as np
import pandas as pd


def engineer_features(df_raw):
    """
    Computes time-context, net surplus, uncertainty scores, and operational states.
    """
    df = df_raw.copy()

    # Ensure timestamp is datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        df["timestamp"] = pd.date_range("2024-01-01", periods=len(df), freq="h")

    # Time context
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["is_daytime"] = ((df["hour"] >= 6) & (df["hour"] <= 18)).astype(int)
    df["is_peak_load"] = (
        ((df["hour"] >= 7) & (df["hour"] <= 10)) |
        ((df["hour"] >= 18) & (df["hour"] <= 22))
    ).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Clip power values to non-negative
    for col in ["solar_power_q10", "solar_power_q50", "solar_power_q90",
                "wind_power_q10", "wind_power_q50", "wind_power_q90"]:
        df[col] = df[col].clip(lower=0.0)

    # Renewable supply features
    df["total_gen_q50"] = df["solar_power_q50"] + df["wind_power_q50"]
    df["total_gen_q10"] = df["solar_power_q10"] + df["wind_power_q10"]
    df["total_gen_q90"] = df["solar_power_q90"] + df["wind_power_q90"]

    df["solar_uncertainty"] = (df["solar_power_q90"] - df["solar_power_q10"]).clip(lower=0.0)
    df["wind_uncertainty"] = (df["wind_power_q90"] - df["wind_power_q10"]).clip(lower=0.0)
    df["load_uncertainty"] = (df["load_demand_q90"] - df["load_demand_q10"]).clip(lower=0.0)

    eps = 1e-2
    df["solar_unc_score"] = (df["solar_uncertainty"] / (df["solar_power_q50"] + eps)).replace([np.inf, -np.inf], 0.0).clip(0, 50)
    df["wind_unc_score"] = (df["wind_uncertainty"] / (df["wind_power_q50"] + eps)).replace([np.inf, -np.inf], 0.0).clip(0, 50)
    df["load_unc_score"] = (df["load_uncertainty"] / (df["load_demand_q50"] + eps)).replace([np.inf, -np.inf], 0.0).clip(0, 50)

    df["net_surplus_q50"] = df["total_gen_q50"] - df["load_demand_q50"]
    df["net_surplus_q10"] = df["total_gen_q10"] - df["load_demand_q10"]
    df["surplus_ratio"] = df["net_surplus_q50"] / (df["load_demand_q50"] + eps)
    df["renewable_cover"] = (df["total_gen_q50"] / (df["load_demand_q50"] + eps)).clip(0, 3)

    # Battery states (fallback or map from battery observations)
    if "soh_pct" not in df.columns:
        if "soh_pred_%" in df.columns:
            df["soh_pct"] = df["soh_pred_%"].fillna(85.0)
        else:
            df["soh_pct"] = 85.0

    if "deg_rate" not in df.columns:
        if "deg_rate_pred_smooth" in df.columns:
            df["deg_rate"] = df["deg_rate_pred_smooth"].fillna(0.0)
        else:
            df["deg_rate"] = 0.0

    if "cumulative_fade" not in df.columns:
        if "cumulative_fade_Ahr" in df.columns:
            df["cumulative_fade"] = df["cumulative_fade_Ahr"].fillna(0.0)
        else:
            df["cumulative_fade"] = 0.0

    if "is_anomaly_any" not in df.columns:
        if "is_anomaly" in df.columns:
            df["is_anomaly_any"] = df["is_anomaly"].fillna(0).astype(int)
        else:
            df["is_anomaly_any"] = 0

    N = len(df)
    rng = np.random.default_rng(seed=42)

    # Diesel cost
    if "diesel_cost_per_kwh" not in df.columns:
        df["diesel_cost_per_kwh"] = (12.0 + rng.normal(0, 0.5, N) + 1.5 * df["month_sin"]).clip(9, 16)

    # Grid availability
    if "grid_available" not in df.columns:
        grid_prob = np.where(df["is_daytime"] == 1, 0.55, 0.30)
        df["grid_available"] = (rng.random(N) < grid_prob).astype(int)

    # State of Charge (SOC)
    if "soc_pct" not in df.columns:
        tod_soc = 50 + 25 * np.sin(np.pi * (df["hour"].values - 6) / 16)
        df["soc_pct"] = np.clip(tod_soc + rng.normal(0, 5, N), 10, 100)

    # Cell temperature
    if "cell_temp_c" not in df.columns:
        ambient = 25 + 12 * np.sin(2 * np.pi * (df["month"].values - 3) / 12)
        df["cell_temp_c"] = np.clip(ambient + 5 * (df["soc_pct"].values / 100) + rng.normal(0, 2, N), 15, 55)

    return df
