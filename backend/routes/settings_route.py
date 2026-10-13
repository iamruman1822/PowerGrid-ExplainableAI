"""
Settings Route — PostgreSQL Backed
==================================
Stores and retrieves microgrid system parameters and configurations per user.
"""

import json
from flask import Blueprint, jsonify, request
from database import get_db_connection
from routes.auth import get_user_from_token

settings_bp = Blueprint("settings", __name__)

DEFAULT_SETTINGS = {
    "seq_len": 48,
    "horizon": 24,
    "batch_size": 32,
    "lstm_hidden": 128,
    "physics_penalty": 10.0,
    "eol_threshold": 80,
    "anomaly_z_score": 2.0,
    "nominal_capacity": 2.0,
    "rotor_area": 50.0,
    "panel_area": 200.0,
    "panel_efficiency": 0.18,
    "api_url": "http://localhost:5000",
    "refresh_interval": 30,
}


def _get_auth_user_id():
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1].strip()
        user = get_user_from_token(token)
        if user:
            return user["id"]
    return None


@settings_bp.route("/settings", methods=["GET"])
def get_settings():
    user_id = _get_auth_user_id()
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if user_id:
                cur.execute("SELECT settings FROM user_settings WHERE user_id = %s;", (user_id,))
                row = cur.fetchone()
                if row and row["settings"]:
                    s = row["settings"]
                    if isinstance(s, str):
                        s = json.loads(s)
                    merged = dict(DEFAULT_SETTINGS)
                    merged.update(s)
                    return jsonify({"settings": merged})

            # Fallback to first user or default
            cur.execute("SELECT settings FROM user_settings LIMIT 1;")
            row = cur.fetchone()
            if row and row["settings"]:
                s = row["settings"]
                if isinstance(s, str):
                    s = json.loads(s)
                merged = dict(DEFAULT_SETTINGS)
                merged.update(s)
                return jsonify({"settings": merged})

        return jsonify({"settings": DEFAULT_SETTINGS})
    finally:
        conn.close()


@settings_bp.route("/settings", methods=["POST"])
def save_settings():
    user_id = _get_auth_user_id()
    data = request.get_json(force=True) or {}
    new_settings = data.get("settings", data)

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if not user_id:
                cur.execute("SELECT id FROM users LIMIT 1;")
                first_user = cur.fetchone()
                if first_user:
                    user_id = first_user["id"]

            if user_id:
                cur.execute("""
                    INSERT INTO user_settings (user_id, settings, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (user_id) DO UPDATE
                    SET settings = EXCLUDED.settings, updated_at = NOW();
                """, (user_id, json.dumps(new_settings)))
                conn.commit()

        return jsonify({"message": "Settings saved to PostgreSQL successfully", "settings": new_settings})
    finally:
        conn.close()
