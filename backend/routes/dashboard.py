"""
Dashboard Route
===============
GET /api/dashboard — Aggregated overview for the main dashboard page.
"""

from flask import Blueprint, jsonify
from services.forecast_service import get_latest_forecast, get_forecast_summary
from services.battery_service import get_fleet_overview
from services.dispatch_service import get_sample_explanations

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/dashboard")
def dashboard_overview():
    """Return aggregated dashboard data combining all three modules."""
    forecast = get_latest_forecast()
    forecast_summary = get_forecast_summary()
    battery = get_fleet_overview()
    recent_decisions = get_sample_explanations(limit=5)

    # Compute alerts
    alerts = []
    if battery.get("batteries_at_risk", 0) > 0:
        alerts.append({
            "type": "warning",
            "title": "Battery Health Alert",
            "message": f"{battery['batteries_at_risk']} batteries below 80% SoH threshold",
        })
    if battery.get("total_anomalies", 0) > 0:
        alerts.append({
            "type": "danger",
            "title": "Anomalies Detected",
            "message": f"{battery['total_anomalies']} anomalies detected across fleet",
        })
    if forecast.get("solar", {}).get("q50", 0) < 1.0:
        alerts.append({
            "type": "info",
            "title": "Low Solar Output",
            "message": "Current solar generation below 1 kW — cloud cover likely",
        })

    return jsonify({
        "forecast": forecast,
        "forecast_summary": forecast_summary,
        "battery": battery,
        "recent_decisions": recent_decisions,
        "alerts": alerts,
    })
