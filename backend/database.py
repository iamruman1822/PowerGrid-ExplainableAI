"""
Database Module — PostgreSQL
=============================
Handles PostgreSQL connection, schema migrations, and initial data seeding
for XplainableAI platform.
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from werkzeug.security import generate_password_hash
import pandas as pd

# Database configuration with defaults for local PostgreSQL
PG_HOST = os.environ.get("PGHOST", "localhost")
PG_PORT = int(os.environ.get("PGPORT", 5432))
PG_USER = os.environ.get("PGUSER", "postgres")
PG_PASSWORD = os.environ.get("PGPASSWORD", "1234")
PG_DATABASE = os.environ.get("PGDATABASE", "xplainable_ai")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_db_connection():
    """Create and return a new PostgreSQL connection with RealDictCursor."""
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        dbname=PG_DATABASE,
        cursor_factory=RealDictCursor,
    )


def init_db():
    """Initialize schema tables and seed default records if empty."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # 1. Users table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    full_name VARCHAR(255),
                    role VARCHAR(50) DEFAULT 'operator',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    last_login TIMESTAMP WITH TIME ZONE
                );
            """)

            # 2. User settings table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_settings (
                    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                    settings JSONB NOT NULL DEFAULT '{}',
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """)

            # 3. Forecast records table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS forecast_records (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    solar_power_q10 DOUBLE PRECISION,
                    solar_power_q50 DOUBLE PRECISION,
                    solar_power_q90 DOUBLE PRECISION,
                    wind_power_q10 DOUBLE PRECISION,
                    wind_power_q50 DOUBLE PRECISION,
                    wind_power_q90 DOUBLE PRECISION,
                    load_demand_q10 DOUBLE PRECISION,
                    load_demand_q50 DOUBLE PRECISION,
                    load_demand_q90 DOUBLE PRECISION
                );
                CREATE INDEX IF NOT EXISTS idx_forecast_timestamp ON forecast_records(timestamp);
            """)

            # 4. Battery fleet table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS battery_fleet (
                    battery VARCHAR(50) PRIMARY KEY,
                    total_cycles INTEGER,
                    initial_capacity_ahr DOUBLE PRECISION,
                    final_actual_cap_ahr DOUBLE PRECISION,
                    final_pred_cap_ahr DOUBLE PRECISION,
                    soh_actual_pct DOUBLE PRECISION,
                    soh_predicted_pct DOUBLE PRECISION,
                    rul_predicted_cycles VARCHAR(50),
                    rul_actual_cycles VARCHAR(50),
                    avg_deg_rate_ahr_cycle DOUBLE PRECISION,
                    anomaly_count INTEGER,
                    mae_ahr DOUBLE PRECISION,
                    rmse_ahr DOUBLE PRECISION
                );
            """)

            # 5. Dispatch history table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS dispatch_history (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                    decision_mode VARCHAR(50) NOT NULL,
                    confidence DOUBLE PRECISION,
                    recommendation TEXT,
                    input_features JSONB,
                    top_drivers JSONB
                );
            """)

            conn.commit()

        # Seed data
        _seed_users(conn)
        _seed_battery_fleet(conn)
        _seed_forecast_records(conn)

        print("[INFO] PostgreSQL database initialized and verified successfully.")
    finally:
        conn.close()


def _seed_users(conn):
    """Seed default administrator and demo user accounts."""
    with conn.cursor() as cur:
        default_users = [
            ("admin@xplainable.ai", "admin123", "Lead Grid Operator", "admin"),
            ("demo@xplainable.ai", "demo123", "Microgrid Field Analyst", "operator"),
        ]
        for email, password, full_name, role in default_users:
            cur.execute("SELECT id FROM users WHERE email = %s;", (email,))
            row = cur.fetchone()
            if not row:
                pw_hash = generate_password_hash(password)
                cur.execute("""
                    INSERT INTO users (email, password_hash, full_name, role)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id;
                """, (email, pw_hash, full_name, role))
                user_id = cur.fetchone()["id"]
                # Insert default settings
                default_settings = {
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
                import json
                cur.execute("""
                    INSERT INTO user_settings (user_id, settings)
                    VALUES (%s, %s)
                    ON CONFLICT (user_id) DO NOTHING;
                """, (user_id, json.dumps(default_settings)))
        conn.commit()


def _seed_battery_fleet(conn):
    """Seed battery_fleet table from battery_full_summary.csv if empty."""
    bat_csv = os.path.join(
        ROOT_DIR, "ZerodayExplorers_BatteryDegradation", "outputs", "data", "battery_full_summary.csv"
    )
    if not os.path.exists(bat_csv):
        return

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS count FROM battery_fleet;")
        count = cur.fetchone()["count"]
        if count == 0:
            df = pd.read_csv(bat_csv)
            records = []
            for _, r in df.iterrows():
                records.append((
                    str(r.get("battery", "")),
                    int(r.get("total_cycles", 0)) if pd.notna(r.get("total_cycles")) else 0,
                    float(r.get("initial_capacity_Ahr", 0)) if pd.notna(r.get("initial_capacity_Ahr")) else None,
                    float(r.get("final_actual_cap_Ahr", 0)) if pd.notna(r.get("final_actual_cap_Ahr")) else None,
                    float(r.get("final_pred_cap_Ahr", 0)) if pd.notna(r.get("final_pred_cap_Ahr")) else None,
                    float(r.get("soh_actual_%", 0)) if pd.notna(r.get("soh_actual_%")) else None,
                    float(r.get("soh_predicted_%", 0)) if pd.notna(r.get("soh_predicted_%")) else None,
                    str(r.get("rul_predicted_cycles", "")) if pd.notna(r.get("rul_predicted_cycles")) else None,
                    str(r.get("rul_actual_cycles", "")) if pd.notna(r.get("rul_actual_cycles")) else None,
                    float(r.get("avg_deg_rate_Ahr_cycle", 0)) if pd.notna(r.get("avg_deg_rate_Ahr_cycle")) else None,
                    int(r.get("anomaly_count", 0)) if pd.notna(r.get("anomaly_count")) else 0,
                    float(r.get("mae_Ahr", 0)) if pd.notna(r.get("mae_Ahr")) else None,
                    float(r.get("rmse_Ahr", 0)) if pd.notna(r.get("rmse_Ahr")) else None,
                ))
            execute_values(
                cur,
                """
                INSERT INTO battery_fleet (
                    battery, total_cycles, initial_capacity_ahr, final_actual_cap_ahr,
                    final_pred_cap_ahr, soh_actual_pct, soh_predicted_pct,
                    rul_predicted_cycles, rul_actual_cycles, avg_deg_rate_ahr_cycle,
                    anomaly_count, mae_ahr, rmse_ahr
                ) VALUES %s ON CONFLICT (battery) DO NOTHING;
                """,
                records
            )
            conn.commit()
            print(f"[INFO] Seeded {len(records)} battery fleet records into PostgreSQL.")


def _seed_forecast_records(conn):
    """Seed forecast_records table from bihar_predictions.csv if empty."""
    fc_csv = os.path.join(ROOT_DIR, "CNN+LSTM-FORECASTER", "data", "bihar_predictions.csv")
    if not os.path.exists(fc_csv):
        return

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS count FROM forecast_records;")
        count = cur.fetchone()["count"]
        if count == 0:
            df = pd.read_csv(fc_csv)
            # Clip negatives for physics realism
            for col in ["solar_power_q10", "solar_power_q50", "solar_power_q90",
                        "wind_power_q10", "wind_power_q50", "wind_power_q90"]:
                if col in df.columns:
                    df[col] = df[col].clip(lower=0)

            records = []
            for _, r in df.iterrows():
                records.append((
                    str(r.get("timestamp")),
                    float(r.get("solar_power_q10", 0)),
                    float(r.get("solar_power_q50", 0)),
                    float(r.get("solar_power_q90", 0)),
                    float(r.get("wind_power_q10", 0)),
                    float(r.get("wind_power_q50", 0)),
                    float(r.get("wind_power_q90", 0)),
                    float(r.get("load_demand_q10", 0)),
                    float(r.get("load_demand_q50", 0)),
                    float(r.get("load_demand_q90", 0)),
                ))

            # Batch insert in chunks of 1000
            chunk_size = 1000
            for i in range(0, len(records), chunk_size):
                chunk = records[i:i + chunk_size]
                execute_values(
                    cur,
                    """
                    INSERT INTO forecast_records (
                        timestamp, solar_power_q10, solar_power_q50, solar_power_q90,
                        wind_power_q10, wind_power_q50, wind_power_q90,
                        load_demand_q10, load_demand_q50, load_demand_q90
                    ) VALUES %s;
                    """,
                    chunk
                )
            conn.commit()
            print(f"[INFO] Seeded {len(records)} forecast records into PostgreSQL.")
