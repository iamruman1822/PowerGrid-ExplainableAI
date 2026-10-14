"""
Microgrid Dispatch Classifier - Constants and Schema
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")

FORECAST_CSV = os.path.join(DATA_DIR, "bihar_predictions.csv")
BATTERY_CSV = os.path.join(DATA_DIR, "nasa_2yr_all_batteries.csv")
MODEL_PATH = os.path.join(OUT_DIR, "dispatch_engine.ubj")

CLASS_NAMES = ["CONSERVATIVE", "MODERATE", "AGGRESSIVE"]

FEATURE_COLS = [
    "solar_power_q10", "solar_power_q50", "solar_power_q90",
    "wind_power_q10", "wind_power_q50", "wind_power_q90",
    "load_demand_q10", "load_demand_q50", "load_demand_q90",
    "total_gen_q50", "net_surplus_q50", "net_surplus_q10",
    "surplus_ratio", "renewable_cover",
    "solar_uncertainty", "wind_uncertainty", "load_uncertainty",
    "solar_unc_score", "wind_unc_score", "load_unc_score",
    "soh_pct", "soc_pct", "deg_rate", "cumulative_fade",
    "cell_temp_c", "is_anomaly_any",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "is_daytime", "is_peak_load",
    "diesel_cost_per_kwh", "grid_available",
]

FEATURE_LABELS = {
    "solar_power_q50": "Solar Power (median kW)",
    "solar_power_q90": "Solar Power (optimistic kW)",
    "solar_power_q10": "Solar Power (pessimistic kW)",
    "wind_power_q50": "Wind Power (median kW)",
    "wind_power_q90": "Wind Power (optimistic kW)",
    "wind_power_q10": "Wind Power (pessimistic kW)",
    "load_demand_q50": "Load Demand (median kW)",
    "load_demand_q90": "Load Demand (high kW)",
    "load_demand_q10": "Load Demand (low kW)",
    "total_gen_q50": "Total Renewable Gen (kW)",
    "net_surplus_q50": "Net Energy Surplus (kW)",
    "net_surplus_q10": "Net Surplus (pessimistic kW)",
    "surplus_ratio": "Surplus Ratio",
    "renewable_cover": "Renewable Coverage Factor",
    "solar_uncertainty": "Solar Forecast Uncertainty (kW)",
    "wind_uncertainty": "Wind Forecast Uncertainty (kW)",
    "load_uncertainty": "Load Forecast Uncertainty (kW)",
    "solar_unc_score": "Solar Uncertainty Score",
    "wind_unc_score": "Wind Uncertainty Score",
    "load_unc_score": "Load Uncertainty Score",
    "soh_pct": "Battery State of Health (%)",
    "soc_pct": "Battery State of Charge (%)",
    "deg_rate": "Battery Degradation Rate",
    "cumulative_fade": "Cumulative Capacity Fade (Ah)",
    "cell_temp_c": "Battery Cell Temp (C)",
    "is_anomaly_any": "Battery Anomaly Flag",
    "hour_sin": "Time of Day (sin)",
    "hour_cos": "Time of Day (cos)",
    "month_sin": "Season (sin)",
    "month_cos": "Season (cos)",
    "is_daytime": "Is Daytime",
    "is_peak_load": "Is Peak Load",
    "diesel_cost_per_kwh": "Diesel Cost (Rs/kWh)",
    "grid_available": "Grid Available",
}

OPERATOR_DESC = {
    "solar_power_q50": "solar generation (median)",
    "solar_power_q90": "solar generation (optimistic)",
    "solar_power_q10": "solar generation (pessimistic)",
    "wind_power_q50": "wind power output",
    "wind_power_q90": "wind power (optimistic)",
    "wind_power_q10": "wind power (pessimistic)",
    "load_demand_q50": "village demand (median)",
    "load_demand_q90": "village demand (high estimate)",
    "load_demand_q10": "village demand (low estimate)",
    "total_gen_q50": "total renewable generation",
    "net_surplus_q50": "net surplus energy",
    "net_surplus_q10": "worst-case surplus",
    "surplus_ratio": "surplus to load ratio",
    "renewable_cover": "renewable coverage",
    "solar_uncertainty": "solar forecast uncertainty",
    "wind_uncertainty": "wind forecast uncertainty",
    "load_uncertainty": "demand forecast uncertainty",
    "solar_unc_score": "solar uncertainty score",
    "wind_unc_score": "wind uncertainty score",
    "load_unc_score": "demand uncertainty score",
    "soh_pct": "battery state-of-health",
    "soc_pct": "battery state-of-charge",
    "deg_rate": "battery degradation speed",
    "cumulative_fade": "cumulative battery fade",
    "cell_temp_c": "battery cell temperature",
    "is_anomaly_any": "battery anomaly state",
    "hour_sin": "time of day",
    "hour_cos": "time of day",
    "month_sin": "seasonality",
    "month_cos": "seasonality",
    "is_daytime": "daytime window",
    "is_peak_load": "peak demand window",
    "diesel_cost_per_kwh": "diesel generator cost",
    "grid_available": "DISCOM grid status",
}

RECOMMENDATIONS = {
    0: "Limit battery discharge immediately. Bring diesel generator online to cover deficit. Curtail high-power deferrable loads until battery recovers.",
    1: "Run in balanced mode. Use solar and wind to charge battery when generation > demand. Draw modestly from battery during minor shortfalls.",
    2: "Maximize renewable usage. Run heavy deferrable loads (irrigation pumps, community mills) now. Top up battery to full capacity and defer diesel usage.",
}
