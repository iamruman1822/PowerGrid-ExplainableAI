"""
Live Inference & Scenario Demonstration Module
Runs live scenarios through the trained model and prints plain-language operator reports.
"""

import numpy as np
import pandas as pd
from .constants import FEATURE_COLS, CLASS_NAMES, OPERATOR_DESC, RECOMMENDATIONS


def build_scenario_feature_row(**kwargs):
    """
    Builds a single-row feature DataFrame from operational parameters.
    """
    eps = 1e-3
    solar_q10 = max(float(kwargs.get("solar_q10", 0)), 0)
    solar_q50 = max(float(kwargs.get("solar_q50", 0)), 0)
    solar_q90 = max(float(kwargs.get("solar_q90", 0)), 0)
    wind_q10 = max(float(kwargs.get("wind_q10", 0)), 0)
    wind_q50 = max(float(kwargs.get("wind_q50", 0)), 0)
    wind_q90 = max(float(kwargs.get("wind_q90", 0)), 0)
    load_q10 = float(kwargs.get("load_q10", 40))
    load_q50 = float(kwargs.get("load_q50", 60))
    load_q90 = float(kwargs.get("load_q90", 80))
    soh_pct = float(kwargs.get("soh_pct", 85))
    soc_pct = float(kwargs.get("soc_pct", 60))
    cell_temp_c = float(kwargs.get("cell_temp_c", 35))
    hour = int(kwargs.get("hour", 14))
    month = int(kwargs.get("month", 6))
    diesel_cost = float(kwargs.get("diesel_cost", 12.0))
    grid_available = int(kwargs.get("grid_available", 0))
    deg_rate = float(kwargs.get("deg_rate", 0.001))
    cumulative_fade = float(kwargs.get("cumulative_fade", 0.1))
    is_anomaly = int(kwargs.get("is_anomaly", 0))

    total_gen = solar_q50 + wind_q50
    total_q10 = solar_q10 + wind_q10
    surplus = total_gen - load_q50
    surp_q10 = total_q10 - load_q90

    row = {
        "solar_power_q10": solar_q10,
        "solar_power_q50": solar_q50,
        "solar_power_q90": solar_q90,
        "wind_power_q10": wind_q10,
        "wind_power_q50": wind_q50,
        "wind_power_q90": wind_q90,
        "load_demand_q10": load_q10,
        "load_demand_q50": load_q50,
        "load_demand_q90": load_q90,
        "total_gen_q50": total_gen,
        "net_surplus_q50": surplus,
        "net_surplus_q10": surp_q10,
        "surplus_ratio": surplus / (load_q50 + eps),
        "renewable_cover": min(total_gen / (load_q50 + eps), 3.0),
        "solar_uncertainty": solar_q90 - solar_q10,
        "wind_uncertainty": wind_q90 - wind_q10,
        "load_uncertainty": load_q90 - load_q10,
        "solar_unc_score": (solar_q90 - solar_q10) / (solar_q50 + eps),
        "wind_unc_score": (wind_q90 - wind_q10) / (wind_q50 + eps),
        "load_unc_score": (load_q90 - load_q10) / (load_q50 + eps),
        "soh_pct": soh_pct,
        "soc_pct": soc_pct,
        "deg_rate": deg_rate,
        "cumulative_fade": cumulative_fade,
        "cell_temp_c": cell_temp_c,
        "is_anomaly_any": is_anomaly,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "is_daytime": int(6 <= hour <= 18),
        "is_peak_load": int((7 <= hour <= 10) or (18 <= hour <= 22)),
        "diesel_cost_per_kwh": diesel_cost,
        "grid_available": grid_available,
    }

    return pd.DataFrame([row])[FEATURE_COLS]


def run_scenario_demo(model, explainer):
    """
    Runs demonstration of 3 distinct operational situations.
    """
    scenarios = [
        {
            "name": "Scenario 1: Peak Monsoon Cloud Cover & Battery Stress",
            "context": "Low solar (1.5 kW), high village load (75 kW), battery SOC at 21%, cell temp at 48 C.",
            "params": {
                "solar_q10": 0.5, "solar_q50": 1.5, "solar_q90": 3.0,
                "wind_q10": 0.2, "wind_q50": 0.5, "wind_q90": 1.0,
                "load_q10": 60, "load_q50": 75, "load_q90": 90,
                "soc_pct": 21.0, "soh_pct": 72.0, "cell_temp_c": 48.0,
                "hour": 14, "month": 7, "diesel_cost": 14.5, "grid_available": 0,
            }
        },
        {
            "name": "Scenario 2: Sunny Afternoon Renewable Surplus",
            "context": "Strong solar (9.2 kW), healthy battery (88% SOH, 75% SOC), low diesel cost.",
            "params": {
                "solar_q10": 7.5, "solar_q50": 9.2, "solar_q90": 11.5,
                "wind_q10": 0.5, "wind_q50": 1.2, "wind_q90": 2.0,
                "load_q10": 30, "load_q50": 42, "load_q90": 55,
                "soc_pct": 75.0, "soh_pct": 88.0, "cell_temp_c": 32.0,
                "hour": 13, "month": 5, "diesel_cost": 11.0, "grid_available": 1,
            }
        }
    ]

    print("\n" + "=" * 68)
    print("   LIVE INFERENCE & OPERATOR EXPLAINABILITY DEMONSTRATIONS")
    print("=" * 68)

    for sc in scenarios:
        X_live = build_scenario_feature_row(**sc["params"])
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_live)[0]
        else:
            import xgboost as xgb
            proba = model.predict(xgb.DMatrix(X_live))[0]

        pred_cls = int(np.argmax(proba))
        conf = float(proba[pred_cls])

        sv = explainer(X_live)
        sv_pred = sv[0, :, pred_cls].values
        top3_idx = np.argsort(np.abs(sv_pred))[-3:][::-1]

        print(f"\n>> {sc['name']}")
        print(f"   Context       : {sc['context']}")
        print(f"   Decision      : {CLASS_NAMES[pred_cls]} (Confidence: {conf*100:.1f}%)")
        print("   Probabilities : " + ", ".join([f"{CLASS_NAMES[i]}: {proba[i]*100:.1f}%" for i in range(3)]))
        print("   Top Drivers   :")
        for rank, fi in enumerate(top3_idx, 1):
            feat = FEATURE_COLS[fi]
            readable = OPERATOR_DESC.get(feat, feat)
            shap_val = sv_pred[fi]
            direction = "pushed TOWARD" if shap_val > 0 else "pushed AWAY FROM"
            print(f"      {rank}. {readable:<30}: {shap_val:+.4f} ({direction} {CLASS_NAMES[pred_cls]})")
        print(f"   Actionable Recommendation : {RECOMMENDATIONS[pred_cls]}")
