"""
Forecast Route
==============
GET /api/forecast — 24h energy forecast data with quantile bands.
GET /api/forecast/summary — Aggregate stats.
"""

from flask import Blueprint, jsonify, request
from services.forecast_service import get_forecast_data, get_forecast_summary

forecast_bp = Blueprint("forecast", __name__)


@forecast_bp.route("/forecast")
def forecast_data():
    """Return forecast time series. Optional ?limit=N query param."""
    limit = request.args.get("limit", None, type=int)
    data = get_forecast_data(limit=limit)
    return jsonify({"data": data, "total": len(data)})


@forecast_bp.route("/forecast/summary")
def forecast_summary():
    """Return forecast summary statistics."""
    return jsonify(get_forecast_summary())
