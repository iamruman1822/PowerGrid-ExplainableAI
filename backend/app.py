"""
XplainableAI — Flask Backend
=============================
Main application entry point. Registers all route blueprints and
configures CORS for the React frontend dev server.
"""

import os
from flask import Flask, send_from_directory
from flask_cors import CORS

from routes.dashboard import dashboard_bp
from routes.forecast import forecast_bp
from routes.battery import battery_bp
from routes.dispatch import dispatch_bp
from routes.explainability import explainability_bp
from routes.auth import auth_bp
from routes.settings_route import settings_bp
from database import init_db


def create_app():
    # Initialize PostgreSQL schema & seed data
    try:
        init_db()
    except Exception as e:
        print(f"[WARN] PostgreSQL init failed: {e}")

    app = Flask(__name__, static_folder=None)
    app.config["SECRET_KEY"] = "xplainable-ai-dev-key"

    # Allow React dev server (Vite default port)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # ── Register blueprints ──────────────────────────────────────────────
    app.register_blueprint(dashboard_bp,      url_prefix="/api")
    app.register_blueprint(forecast_bp,       url_prefix="/api")
    app.register_blueprint(battery_bp,        url_prefix="/api")
    app.register_blueprint(dispatch_bp,       url_prefix="/api")
    app.register_blueprint(explainability_bp, url_prefix="/api")
    app.register_blueprint(auth_bp,           url_prefix="/api")
    app.register_blueprint(settings_bp,       url_prefix="/api")

    # ── Health check ─────────────────────────────────────────────────────
    @app.route("/api/health")
    def health():
        return {
            "status": "ok",
            "message": "XplainableAI backend is running",
            "database": "postgresql (xplainable_ai)",
        }

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)
