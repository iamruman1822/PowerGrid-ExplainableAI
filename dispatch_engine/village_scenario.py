"""
village_scenario.py
-------------------
Runs a specific, richly-described real-world Bihar village scenario
through the trained dispatch engine and prints a full operator report.
"""

import os, sys
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import textwrap

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.join(BASE_DIR, "outputs")
MODEL_PATH = os.path.join(OUT_DIR, "dispatch_engine.ubj")

CLASS_NAMES = ["CONSERVATIVE", "MODERATE", "AGGRESSIVE"]

FEATURE_COLS = [
    "solar_power_q10", "solar_power_q50", "solar_power_q90",
    "wind_power_q10",  "wind_power_q50",  "wind_power_q90",
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
    "solar_power_q50"    : "Solar Power – median (kW)",
    "solar_power_q90"    : "Solar Power – optimistic (kW)",
    "solar_power_q10"    : "Solar Power – pessimistic (kW)",
    "wind_power_q50"     : "Wind Power – median (kW)",
    "wind_power_q90"     : "Wind Power – optimistic (kW)",
    "wind_power_q10"     : "Wind Power – pessimistic (kW)",
    "load_demand_q50"    : "Load Demand – median (kW)",
    "load_demand_q90"    : "Load Demand – high estimate (kW)",
    "load_demand_q10"    : "Load Demand – low estimate (kW)",
    "total_gen_q50"      : "Total Renewable Generation (kW)",
    "net_surplus_q50"    : "Net Energy Surplus (kW)",
    "net_surplus_q10"    : "Net Surplus – pessimistic (kW)",
    "surplus_ratio"      : "Surplus Ratio (fraction of load)",
    "renewable_cover"    : "Renewable Cover Factor",
    "solar_uncertainty"  : "Solar Forecast Uncertainty (kW)",
    "wind_uncertainty"   : "Wind Forecast Uncertainty (kW)",
    "load_uncertainty"   : "Load Forecast Uncertainty (kW)",
    "solar_unc_score"    : "Solar Uncertainty Score",
    "wind_unc_score"     : "Wind Uncertainty Score",
    "load_unc_score"     : "Load Uncertainty Score",
    "soh_pct"            : "Battery State of Health (%)",
    "soc_pct"            : "Battery State of Charge (%)",
    "deg_rate"           : "Battery Degradation Rate",
    "cumulative_fade"    : "Cumulative Capacity Fade (Ah)",
    "cell_temp_c"        : "Battery Cell Temperature (°C)",
    "is_anomaly_any"     : "Battery Anomaly Detected",
    "hour_sin"           : "Time of Day – sin",
    "hour_cos"           : "Time of Day – cos",
    "month_sin"          : "Season – sin",
    "month_cos"          : "Season – cos",
    "is_daytime"         : "Is Daytime",
    "is_peak_load"       : "Is Peak Load Period",
    "diesel_cost_per_kwh": "Diesel Cost (Rs/kWh)",
    "grid_available"     : "Grid Power Available",
}

OPERATOR_DESC = {
    "solar_power_q50"    : "solar generation (best estimate)",
    "solar_power_q90"    : "solar generation (optimistic)",
    "solar_power_q10"    : "solar generation (pessimistic)",
    "wind_power_q50"     : "wind power output",
    "wind_power_q90"     : "wind power (optimistic)",
    "wind_power_q10"     : "wind power (pessimistic)",
    "load_demand_q50"    : "village power demand",
    "load_demand_q90"    : "village demand (high estimate)",
    "load_demand_q10"    : "village demand (low estimate)",
    "total_gen_q50"      : "total solar + wind generation",
    "net_surplus_q50"    : "surplus energy after meeting demand",
    "net_surplus_q10"    : "surplus in worst-case scenario",
    "surplus_ratio"      : "how much extra energy is available",
    "renewable_cover"    : "how well renewables cover the load",
    "solar_uncertainty"  : "uncertainty in solar forecast",
    "wind_uncertainty"   : "uncertainty in wind forecast",
    "load_uncertainty"   : "uncertainty in demand forecast",
    "solar_unc_score"    : "solar forecast reliability",
    "wind_unc_score"     : "wind forecast reliability",
    "load_unc_score"     : "demand forecast reliability",
    "soh_pct"            : "battery health level",
    "soc_pct"            : "battery charge level",
    "deg_rate"           : "battery degradation speed",
    "cumulative_fade"    : "total battery wear so far",
    "cell_temp_c"        : "battery cell temperature",
    "is_anomaly_any"     : "battery fault/anomaly status",
    "hour_sin"           : "time of day",
    "hour_cos"           : "time of day",
    "month_sin"          : "season of the year",
    "month_cos"          : "season of the year",
    "is_daytime"         : "daytime indicator",
    "is_peak_load"       : "peak demand period",
    "diesel_cost_per_kwh": "diesel price",
    "grid_available"     : "grid power availability",
}

RECOMMENDATIONS = {
    0: "Limit battery discharge immediately. If diesel generator is available, "
       "bring it online to cover the deficit. Do NOT run heavy loads (motors, pumps) "
       "until battery recovers above 40%.",
    1: "Run as normal. Use solar and wind to charge battery when generation > demand. "
       "Draw from battery modestly during shortfalls. Keep an eye on diesel reserve.",
    2: "This is a good window! Maximise renewable usage now. Run heavy loads (irrigation "
       "pump, grain mill) during this period. Top up battery to full. Consider deferring "
       "diesel usage — it is not needed right now.",
}

# ── load model ────────────────────────────────────────────────────────────────
try:
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
except Exception:
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
explainer = shap.TreeExplainer(model)

# ── build feature vector ──────────────────────────────────────────────────────
def build_features(solar_q10, solar_q50, solar_q90,
                   wind_q10,  wind_q50,  wind_q90,
                   load_q10,  load_q50,  load_q90,
                   soh_pct, soc_pct, cell_temp_c,
                   hour, month,
                   diesel_cost, grid_available,
                   deg_rate=0.0, cumulative_fade=0.0, is_anomaly=0):
    eps = 1e-3
    solar_q10 = max(solar_q10, 0); solar_q50 = max(solar_q50, 0); solar_q90 = max(solar_q90, 0)
    wind_q10  = max(wind_q10,  0); wind_q50  = max(wind_q50,  0); wind_q90  = max(wind_q90,  0)
    total_gen = solar_q50 + wind_q50
    total_q10 = solar_q10 + wind_q10
    surplus   = total_gen - load_q50
    surp_q10  = total_q10 - load_q90

    row = {
        "solar_power_q10"   : solar_q10,
        "solar_power_q50"   : solar_q50,
        "solar_power_q90"   : solar_q90,
        "wind_power_q10"    : wind_q10,
        "wind_power_q50"    : wind_q50,
        "wind_power_q90"    : wind_q90,
        "load_demand_q10"   : load_q10,
        "load_demand_q50"   : load_q50,
        "load_demand_q90"   : load_q90,
        "total_gen_q50"     : total_gen,
        "net_surplus_q50"   : surplus,
        "net_surplus_q10"   : surp_q10,
        "surplus_ratio"     : surplus / (load_q50 + eps),
        "renewable_cover"   : min(total_gen / (load_q50 + eps), 3.0),
        "solar_uncertainty" : solar_q90 - solar_q10,
        "wind_uncertainty"  : wind_q90  - wind_q10,
        "load_uncertainty"  : load_q90  - load_q10,
        "solar_unc_score"   : (solar_q90 - solar_q10) / (solar_q50 + eps),
        "wind_unc_score"    : (wind_q90  - wind_q10)  / (wind_q50  + eps),
        "load_unc_score"    : (load_q90  - load_q10)  / (load_q50  + eps),
        "soh_pct"           : soh_pct,
        "soc_pct"           : soc_pct,
        "deg_rate"          : deg_rate,
        "cumulative_fade"   : cumulative_fade,
        "cell_temp_c"       : cell_temp_c,
        "is_anomaly_any"    : is_anomaly,
        "hour_sin"          : np.sin(2 * np.pi * hour  / 24),
        "hour_cos"          : np.cos(2 * np.pi * hour  / 24),
        "month_sin"         : np.sin(2 * np.pi * month / 12),
        "month_cos"         : np.cos(2 * np.pi * month / 12),
        "is_daytime"        : int(6 <= hour <= 18),
        "is_peak_load"      : int((7 <= hour <= 10) or (18 <= hour <= 22)),
        "diesel_cost_per_kwh": diesel_cost,
        "grid_available"    : grid_available,
    }
    return pd.DataFrame([row])[FEATURE_COLS]


def run_scenario(scenario: dict):
    """Runs inference on a scenario dict and prints a full operator report."""

    X = build_features(**{k: v for k, v in scenario.items()
                          if k not in ("name", "story", "time_label")})

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
    else:
        dm = xgb.DMatrix(X)
        proba = model.predict(dm)[0]
    pred_cls = int(np.argmax(proba))
    conf     = float(proba[pred_cls])

    sv       = explainer(X)               # shap Explanation object
    sv_pred  = sv[0, :, pred_cls].values  # SHAP for predicted class

    # top-5 drivers
    top5_idx = np.argsort(np.abs(sv_pred))[-5:][::-1]

    feature_display_names = [FEATURE_LABELS.get(c, c) for c in FEATURE_COLS]

    W = 72
    bar = "=" * W

    print(f"\n{bar}")
    print(f"  VILLAGE SCENARIO: {scenario['name']}")
    print(f"  Time: {scenario['time_label']}")
    print(bar)

    # Story / context
    print()
    for line in textwrap.wrap(scenario["story"], width=W - 2):
        print(f"  {line}")

    # Sensor snapshot
    print(f"\n{'-' * W}")
    print("  SENSOR SNAPSHOT")
    print(f"{'-' * W}")
    sensor_rows = [
        ("Solar output (best estimate)",  f"{scenario['solar_q50']:.1f} kW"),
        ("Wind output",                   f"{scenario['wind_q50']:.1f} kW"),
        ("Village load demand",           f"{scenario['load_q50']:.1f} kW"),
        ("Renewable cover",               f"{(scenario['solar_q50']+scenario['wind_q50'])/scenario['load_q50']*100:.1f}% of load"),
        ("Battery State of Health",       f"{scenario['soh_pct']:.1f}%"),
        ("Battery State of Charge",       f"{scenario['soc_pct']:.1f}%"),
        ("Battery cell temperature",      f"{scenario['cell_temp_c']:.1f} °C"),
        ("Diesel cost",                   f"Rs {scenario['diesel_cost']:.2f}/kWh"),
        ("Grid availability",             "YES" if scenario['grid_available'] else "NO"),
    ]
    for label, val in sensor_rows:
        print(f"  {label:<35} {val}")

    # Decision
    print(f"\n{'-' * W}")
    print("  DISPATCH DECISION")
    print(f"{'-' * W}")

    CLASS_ICONS = {0: "SHIELD", 1: "BALANCE", 2: "ROCKET"}
    icon = CLASS_ICONS[pred_cls]

    print(f"\n  [{icon}]  Decision  : {CLASS_NAMES[pred_cls]}")
    print(f"           Confidence : {conf*100:.1f}%")
    print(f"\n  Class probabilities:")
    bars_max = 40
    for i, name in enumerate(CLASS_NAMES):
        p = proba[i]
        filled = int(p * bars_max)
        bar_str = "#" * filled + "-" * (bars_max - filled)
        print(f"    {name:<15} [{bar_str}] {p*100:.1f}%")

    # Why
    print(f"\n{'-' * W}")
    print("  WHY THIS DECISION WAS MADE  (top 5 SHAP drivers)")
    print(f"{'-' * W}")
    for rank, fi in enumerate(top5_idx, 1):
        feat     = FEATURE_COLS[fi]
        readable = OPERATOR_DESC.get(feat, feat).capitalize()
        val      = X.iloc[0, fi]
        sv_val   = sv_pred[fi]
        direction = "pushed TOWARD" if sv_val > 0 else "pushed AWAY FROM"

        # nice value formatting
        if feat in ("is_daytime", "is_peak_load", "grid_available", "is_anomaly_any"):
            val_str = "YES" if val > 0.5 else "NO"
        elif feat in ("hour_sin", "hour_cos", "month_sin", "month_cos"):
            val_str = f"{val:+.3f}"
        else:
            val_str = f"{val:.2f}"

        shap_bar_len = min(int(abs(sv_val) * 60), 20)
        shap_bar = ("+" if sv_val > 0 else "-") * shap_bar_len

        print(f"\n  {rank}. {readable}")
        print(f"     Value : {val_str}   |   SHAP : {sv_val:+.4f}  [{shap_bar}]")
        print(f"     Effect: {direction} {CLASS_NAMES[pred_cls]}")

    # Recommendation
    print(f"\n{'-' * W}")
    print("  OPERATOR RECOMMENDATION")
    print(f"{'-' * W}")
    for line in textwrap.wrap(RECOMMENDATIONS[pred_cls], width=W - 2):
        print(f"  {line}")

    print(f"\n{'=' * W}\n")
    return pred_cls, conf, proba


# ─────────────────────────────────────────────────────────────────────────────
# THE SCENARIO: Rampur village, Bihar — a complicated Tuesday afternoon
# ─────────────────────────────────────────────────────────────────────────────

scenario = dict(
    name       = "Rampur village, Gaya district — Tuesday 14:30",
    time_label = "June, 14:30 IST (post-monsoon onset, partial cloud cover)",

    story = (
        "It is mid-afternoon in Rampur, a 340-household village 60 km south of Gaya. "
        "The monsoon reached Bihar three days ago. The sky is split — patches of thick "
        "cumulus cloud keep interrupting what would otherwise be strong June sun. "
        "The solar forecast is therefore WIDE: the model thinks output could be anywhere "
        "between 1.8 kW and 10.5 kW. The wind has picked up ahead of the next rain band "
        "and is contributing a reliable 1.6 kW. "
        "Village demand is elevated: a wedding preparation is underway — the community "
        "hall has sound equipment, lighting, and an industrial mixer all running. "
        "Estimated load is 78 kW. "
        "The battery bank was fully charged this morning but the operator noticed Cell-3 "
        "threw a temperature warning at 11:00 — it has been running 2 degrees hotter "
        "than the others all day. SoH has been declining steadily: the fleet is at 76.4%. "
        "The state electricity board DISCOM grid has been up since noon — unusual for "
        "a Tuesday — offering a safety net at Rs 7.20/kWh, but diesel is expensive "
        "at Rs 15.10/kWh after the recent price hike. "
        "What should the operator do?"
    ),

    # Forecaster inputs
    solar_q10 = 1.8,
    solar_q50 = 5.1,
    solar_q90 = 10.5,

    wind_q10  = 1.1,
    wind_q50  = 1.6,
    wind_q90  = 2.4,

    load_q10  = 64.0,
    load_q50  = 78.0,
    load_q90  = 91.0,

    # Battery health (from degradation model)
    soh_pct        = 76.4,    # declining fleet
    soc_pct        = 69.0,    # well charged from morning
    cell_temp_c    = 47.5,    # Cell-3 warning — above normal
    deg_rate       = -0.0082, # moderate degradation
    cumulative_fade = 0.19,   # ~11% capacity lost over lifetime
    is_anomaly     = 1,       # Cell-3 temperature anomaly flagged

    # Contextual
    hour           = 14,
    month          = 6,       # June
    diesel_cost    = 15.10,   # expensive
    grid_available = 1,       # grid is up (unusual but true today)
)

pred_cls, conf, proba = run_scenario(scenario)
