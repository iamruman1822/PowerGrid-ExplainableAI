"""
Dispatch Service
================
Loads the trained XGBoost dispatch engine model and SHAP explainer.
Provides predict_and_explain() for live inference via API.
"""

import os
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(ROOT_DIR, "dispatch_engine", "outputs", "dispatch_engine.ubj")
EXPLANATIONS_CSV = os.path.join(ROOT_DIR, "dispatch_engine", "outputs", "per_sample_explanations.csv")
DISPATCH_OUT_DIR = os.path.join(ROOT_DIR, "dispatch_engine", "outputs")

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

OPERATOR_DESC = {
    "solar_power_q50": "solar generation (best estimate)",
    "solar_power_q90": "solar generation (optimistic)",
    "solar_power_q10": "solar generation (pessimistic)",
    "wind_power_q50": "wind power output",
    "wind_power_q90": "wind power (optimistic)",
    "wind_power_q10": "wind power (pessimistic)",
    "load_demand_q50": "village power demand",
    "load_demand_q90": "village demand (high estimate)",
    "load_demand_q10": "village demand (low estimate)",
    "total_gen_q50": "total solar + wind generation",
    "net_surplus_q50": "surplus energy after meeting demand",
    "net_surplus_q10": "surplus in worst-case scenario",
    "surplus_ratio": "how much extra energy is available",
    "renewable_cover": "how well renewables cover the load",
    "solar_uncertainty": "uncertainty in solar forecast",
    "wind_uncertainty": "uncertainty in wind forecast",
    "load_uncertainty": "uncertainty in demand forecast",
    "solar_unc_score": "solar forecast reliability",
    "wind_unc_score": "wind forecast reliability",
    "load_unc_score": "demand forecast reliability",
    "soh_pct": "battery health level",
    "soc_pct": "battery charge level",
    "deg_rate": "battery degradation speed",
    "cumulative_fade": "total battery wear so far",
    "cell_temp_c": "battery cell temperature",
    "is_anomaly_any": "battery fault/anomaly status",
    "hour_sin": "time of day",
    "hour_cos": "time of day",
    "month_sin": "season of the year",
    "month_cos": "season of the year",
    "is_daytime": "daytime indicator",
    "is_peak_load": "peak demand period",
    "diesel_cost_per_kwh": "diesel price",
    "grid_available": "grid power availability",
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

# Lazy-loaded globals
_model = None
_explainer = None


def _load_model():
    """Lazy-load XGBoost XGBClassifier and SHAP TreeExplainer."""
    global _model, _explainer
    if _model is not None:
        return _model, _explainer

    try:
        import xgboost as xgb
        import shap
        # Load as XGBClassifier — matches how dispatch_engine.py saved it
        clf = xgb.XGBClassifier()
        clf.load_model(MODEL_PATH)
        print(f"[INFO] Dispatch model loaded as XGBClassifier")
        _model = clf
        _explainer = shap.TreeExplainer(clf)
        return _model, _explainer
    except Exception as e:
        print(f"[WARN] Could not load dispatch model: {e}")
        return None, None


def predict_and_explain(params):
    """
    Run dispatch prediction with SHAP explanation.

    Parameters
    ----------
    params : dict with keys:
        solar_q10, solar_q50, solar_q90,
        wind_q10, wind_q50, wind_q90,
        load_q10, load_q50, load_q90,
        soh_pct, soc_pct, cell_temp_c,
        hour, month,
        diesel_cost, grid_available,
        deg_rate (optional), cumulative_fade (optional), is_anomaly (optional)

    Returns
    -------
    dict with decision, confidence, class_probs, explanation, shap_drivers, recommendation
    """
    model, explainer = _load_model()
    if model is None:
        return {"error": "Dispatch model not available"}

    eps = 1e-3
    solar_q10 = max(float(params.get("solar_q10", 0)), 0)
    solar_q50 = max(float(params.get("solar_q50", 0)), 0)
    solar_q90 = max(float(params.get("solar_q90", 0)), 0)
    wind_q10 = max(float(params.get("wind_q10", 0)), 0)
    wind_q50 = max(float(params.get("wind_q50", 0)), 0)
    wind_q90 = max(float(params.get("wind_q90", 0)), 0)
    load_q10 = float(params.get("load_q10", 40))
    load_q50 = float(params.get("load_q50", 60))
    load_q90 = float(params.get("load_q90", 80))
    soh_pct = float(params.get("soh_pct", 85))
    soc_pct = float(params.get("soc_pct", 60))
    cell_temp_c = float(params.get("cell_temp_c", 35))
    hour = int(params.get("hour", 12))
    month = int(params.get("month", 6))
    diesel_cost = float(params.get("diesel_cost", 12.0))
    grid_available = int(params.get("grid_available", 0))
    deg_rate = float(params.get("deg_rate", 0.0))
    cumulative_fade = float(params.get("cumulative_fade", 0.0))
    is_anomaly = int(params.get("is_anomaly", 0))

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
        "renewable_cover": min(total_gen / (load_q50 + eps), 3),
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

    X_live = pd.DataFrame([row])[FEATURE_COLS]
    # XGBClassifier API — returns probabilities directly
    proba = model.predict_proba(X_live)[0]
    pred_cls = int(np.argmax(proba))
    conf = float(proba[pred_cls])

    # SHAP — TreeExplainer on XGBClassifier returns Explanation(n_samples, n_features, n_classes)
    sv = explainer(X_live)
    sv_pred = sv[0, :, pred_cls].values

    # Top 5 drivers
    top5 = np.argsort(np.abs(sv_pred))[-5:][::-1]
    drivers = []
    for rank, fi in enumerate(top5, 1):
        feat = FEATURE_COLS[fi]
        readable = OPERATOR_DESC.get(feat, feat).capitalize()
        val = float(X_live.iloc[0, fi])
        shap_val = float(sv_pred[fi])
        direction = "pushed TOWARD" if shap_val > 0 else "pushed AWAY FROM"

        if feat in ("is_daytime", "is_peak_load", "grid_available", "is_anomaly_any"):
            val_str = "YES" if val > 0.5 else "NO"
        else:
            val_str = f"{val:.2f}"

        drivers.append({
            "rank": rank,
            "feature": readable,
            "feature_key": feat,
            "value": val_str,
            "shap_value": round(shap_val, 4),
            "direction": direction,
            "effect": f"{direction} {CLASS_NAMES[pred_cls]}",
        })

    result = {
        "decision": CLASS_NAMES[pred_cls],
        "confidence": round(conf, 4),
        "class_probs": {CLASS_NAMES[i]: round(float(proba[i]), 4) for i in range(3)},
        "drivers": drivers,
        "recommendation": RECOMMENDATIONS[pred_cls],
        "input_summary": {
            "solar_kw": solar_q50,
            "wind_kw": wind_q50,
            "load_kw": load_q50,
            "soh_pct": soh_pct,
            "soc_pct": soc_pct,
            "cell_temp_c": cell_temp_c,
            "renewable_cover_pct": round(total_gen / (load_q50 + eps) * 100, 1),
        },
    }

    # Save to PostgreSQL dispatch_history
    try:
        import json
        from database import get_db_connection
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO dispatch_history (
                        decision_mode, confidence, recommendation, input_features, top_drivers
                    ) VALUES (%s, %s, %s, %s, %s);
                """, (
                    result["decision"],
                    result["confidence"],
                    result["recommendation"],
                    json.dumps(result["input_summary"]),
                    json.dumps(result["drivers"]),
                ))
                conn.commit()
        finally:
            conn.close()
    except Exception as e:
        print(f"[WARN] Failed to log dispatch to PostgreSQL: {e}")

    return result


def get_sample_explanations(limit=20):
    """Return a sample of pre-computed per-sample explanations."""
    if not os.path.exists(EXPLANATIONS_CSV):
        return []
    df = pd.read_csv(EXPLANATIONS_CSV)
    if limit:
        df = df.head(limit)
    return df.where(df.notna(), None).to_dict(orient="records")


def get_shap_image_path(filename):
    """Return path to a SHAP output image."""
    path = os.path.join(DISPATCH_OUT_DIR, filename)
    if os.path.exists(path):
        return path
    return None
