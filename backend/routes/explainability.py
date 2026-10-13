"""
Explainability Route
====================
GET /api/explain/images/<filename> — Serve SHAP plot images.
GET /api/explain/importance — Feature importance data.
"""

import os
from flask import Blueprint, jsonify, send_file, abort
from services.dispatch_service import get_shap_image_path

explainability_bp = Blueprint("explainability", __name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DISPATCH_OUT = os.path.join(ROOT_DIR, "dispatch_engine", "outputs")
FORECASTER_OUT = os.path.join(ROOT_DIR, "CNN+LSTM-FORECASTER", "outputs")


@explainability_bp.route("/explain/images/<filename>")
def shap_image(filename):
    """Serve SHAP or model output images."""
    # Try dispatch outputs first, then forecaster
    path = get_shap_image_path(filename)
    if path:
        mimetype = "text/html" if filename.endswith(".html") else "image/png"
        return send_file(path, mimetype=mimetype)

    # Try forecaster outputs
    fpath = os.path.join(FORECASTER_OUT, filename)
    if os.path.exists(fpath):
        return send_file(fpath, mimetype="image/png")

    abort(404, description=f"Image {filename} not found")


@explainability_bp.route("/explain/available")
def available_plots():
    """List all available SHAP / model output images."""
    plots = []

    # Dispatch engine outputs
    if os.path.isdir(DISPATCH_OUT):
        for f in sorted(os.listdir(DISPATCH_OUT)):
            if f.endswith((".png", ".html")):
                plots.append({
                    "filename": f,
                    "module": "dispatch_engine",
                    "url": f"/api/explain/images/{f}",
                    "type": "html" if f.endswith(".html") else "image",
                })

    # Forecaster outputs
    if os.path.isdir(FORECASTER_OUT):
        for f in sorted(os.listdir(FORECASTER_OUT)):
            if f.endswith(".png"):
                plots.append({
                    "filename": f,
                    "module": "forecaster",
                    "url": f"/api/explain/images/{f}",
                    "type": "image",
                })

    return jsonify({"plots": plots, "total": len(plots)})
