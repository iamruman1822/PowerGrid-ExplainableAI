"""
Battery Route
=============
GET /api/battery — Per-battery health summary.
GET /api/battery/fleet — Fleet overview.
GET /api/battery/cycles — Per-cycle prediction data.
GET /api/battery/ids — List of battery IDs.
GET /api/battery/plots/<filename> — Serve plot images.
"""

from flask import Blueprint, jsonify, request, send_file, abort
from services.battery_service import (
    get_battery_summary,
    get_fleet_overview,
    get_battery_cycles,
    get_battery_ids,
    get_plot_path,
)

battery_bp = Blueprint("battery", __name__)


@battery_bp.route("/battery")
def battery_summary():
    """Return per-battery health summary."""
    data = get_battery_summary()
    return jsonify({"data": data, "total": len(data)})


@battery_bp.route("/battery/fleet")
def fleet_overview():
    """Return fleet-wide aggregated metrics."""
    return jsonify(get_fleet_overview())


@battery_bp.route("/battery/cycles")
def battery_cycles():
    """Return per-cycle predictions. Optional ?battery=B0025"""
    battery_id = request.args.get("battery", None)
    data = get_battery_cycles(battery_id)
    return jsonify({"data": data, "total": len(data)})


@battery_bp.route("/battery/ids")
def battery_ids():
    """Return list of battery IDs."""
    return jsonify(get_battery_ids())


@battery_bp.route("/battery/plots/<filename>")
def battery_plot(filename):
    """Serve a battery plot image."""
    path = get_plot_path(filename)
    if path is None:
        abort(404, description=f"Plot {filename} not found")
    return send_file(path, mimetype="image/png")
