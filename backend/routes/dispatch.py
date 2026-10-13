"""
Dispatch Route
==============
POST /api/dispatch/predict — Run live dispatch inference + SHAP.
GET  /api/dispatch/explanations — Pre-computed sample explanations.
"""

from flask import Blueprint, jsonify, request
from services.dispatch_service import predict_and_explain, get_sample_explanations

dispatch_bp = Blueprint("dispatch", __name__)


@dispatch_bp.route("/dispatch/predict", methods=["POST"])
def dispatch_predict():
    """
    Run dispatch prediction with SHAP explanation.
    Expects JSON body with sensor readings.
    """
    params = request.get_json(force=True)
    if not params:
        return jsonify({"error": "No input data provided"}), 400

    result = predict_and_explain(params)
    return jsonify(result)


@dispatch_bp.route("/dispatch/explanations")
def dispatch_explanations():
    """Return pre-computed per-sample explanations."""
    limit = request.args.get("limit", 20, type=int)
    data = get_sample_explanations(limit=limit)
    return jsonify({"data": data, "total": len(data)})
