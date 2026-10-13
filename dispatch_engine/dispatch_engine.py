"""
================================================================================
 BIHAR RURAL MICROGRID — XGBoost Energy Dispatch Decision Engine
 with SHAP Explainability
================================================================================

Pipeline overview
-----------------
1.  Load & merge the two upstream model outputs:
      • bihar_predictions.csv   → CNN+LSTM solar/wind/load forecasts
      • nasa_2yr_all_batteries.csv → battery degradation tracker
2.  Engineer features that an energy controller would actually care about.
3.  Generate synthetic labels using physical dispatch rules.
4.  Train an XGBoost classifier with early stopping.
5.  Evaluate fully with classification report + confusion matrix.
6.  SHAP global explanations (beeswarm + importance bar).
7.  SHAP per-sample plain-English explanations for every test prediction.
8.  SHAP waterfall + force plots for one representative from each class.
9.  predict_and_explain() — live inference function.

Dispatch classes
----------------
  0 = CONSERVATIVE  — protect battery, minimise discharge
  1 = MODERATE      — balanced renewable dispatch
  2 = AGGRESSIVE    — maximise renewable usage

Dependencies
------------
    pip install xgboost shap pandas numpy scikit-learn matplotlib seaborn
"""

# ─────────────────────────────────────────────────────────────────────────────
# Standard imports
# ─────────────────────────────────────────────────────────────────────────────
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — saves files instead of showing GUI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import xgboost as xgb
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(BASE_DIR)            # d:\XplainableAi

FORECAST_CSV  = os.path.join(
    ROOT_DIR, "CNN+LSTM-FORECASTER", "data", "bihar_predictions.csv"
)
BATTERY_CSV   = os.path.join(
    ROOT_DIR, "ZerodayExplorers_BatteryDegradation",
    "outputs", "data", "nasa_2yr_all_batteries.csv"
)
OUT_DIR       = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Class labels
CLASS_NAMES   = ["CONSERVATIVE", "MODERATE", "AGGRESSIVE"]
CLASS_COLORS  = ["#4e79a7", "#59a14f", "#f28e2b"]   # blue / green / orange

print("=" * 72)
print("  BIHAR MICROGRID — XGBoost Dispatch Engine + SHAP Explainability")
print("=" * 72)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load upstream model outputs
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/9] Loading upstream model outputs …")

# ── 1a. Forecaster output ────────────────────────────────────────────────────
forecast_df = pd.read_csv(FORECAST_CSV, parse_dates=["timestamp"])
forecast_df = forecast_df.sort_values("timestamp").reset_index(drop=True)

# Clamp negative power to 0 (physics: no negative solar/wind)
for col in ["solar_power_q10", "solar_power_q50", "solar_power_q90",
            "wind_power_q10",  "wind_power_q50",  "wind_power_q90"]:
    forecast_df[col] = forecast_df[col].clip(lower=0)

# ── 1b. Battery degradation output ──────────────────────────────────────────
batt_df = pd.read_csv(BATTERY_CSV, parse_dates=["date"])

# Keep only confirmed (non-forecast) rows that have valid SoH
batt_obs = batt_df[batt_df["is_forecast"] == False].copy()
batt_obs = batt_obs.dropna(subset=["soh_pred_%"])

# Aggregate across batteries per date (median — robust to outliers)
batt_daily = (
    batt_obs
    .groupby("date")
    .agg(
        soh_pct          = ("soh_pred_%",          "median"),
        deg_rate         = ("deg_rate_pred_smooth", "median"),
        cumulative_fade  = ("cumulative_fade_Ahr",  "median"),
        is_anomaly_any   = ("is_anomaly",           "max"),   # any battery alarming?
    )
    .reset_index()
    .rename(columns={"date": "timestamp"})
)
batt_daily["timestamp"] = pd.to_datetime(batt_daily["timestamp"])
batt_daily.fillna(method="ffill", inplace=True)

print(f"   Forecaster rows  : {len(forecast_df):,}  "
      f"({forecast_df['timestamp'].min().date()} → {forecast_df['timestamp'].max().date()})")
print(f"   Battery obs rows : {len(batt_obs):,} across "
      f"{batt_obs['battery'].nunique()} batteries")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Feature engineering
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/9] Engineering features …")

df = forecast_df.copy()

# ── 2a. Time context ─────────────────────────────────────────────────────────
df["hour"]          = df["timestamp"].dt.hour
df["month"]         = df["timestamp"].dt.month
df["is_daytime"]    = ((df["hour"] >= 6) & (df["hour"] <= 18)).astype(int)
df["is_peak_load"]  = (
    ((df["hour"] >= 7) & (df["hour"] <= 10)) |
    ((df["hour"] >= 18) & (df["hour"] <= 22))
).astype(int)
# Cyclical encoding for hour (avoids jump from 23→0)
df["hour_sin"]      = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"]      = np.cos(2 * np.pi * df["hour"] / 24)
df["month_sin"]     = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"]     = np.cos(2 * np.pi * df["month"] / 12)

# ── 2b. Renewable supply features ────────────────────────────────────────────
# Best estimate of total generation
df["total_gen_q50"]  = df["solar_power_q50"] + df["wind_power_q50"]
df["total_gen_q10"]  = df["solar_power_q10"] + df["wind_power_q10"]
df["total_gen_q90"]  = df["solar_power_q90"] + df["wind_power_q90"]

# Uncertainty bands (how wide is the confidence interval?)
df["solar_uncertainty"]  = df["solar_power_q90"] - df["solar_power_q10"]
df["wind_uncertainty"]   = df["wind_power_q90"]  - df["wind_power_q10"]
df["load_uncertainty"]   = df["load_demand_q90"]  - df["load_demand_q10"]

# Normalised uncertainty score (fraction of median estimate)
eps = 1e-3
df["solar_unc_score"] = df["solar_uncertainty"] / (df["solar_power_q50"] + eps)
df["wind_unc_score"]  = df["wind_uncertainty"]  / (df["wind_power_q50"]  + eps)
df["load_unc_score"]  = df["load_uncertainty"]  / (df["load_demand_q50"] + eps)

# Net surplus: generation minus demand (negative = deficit)
df["net_surplus_q50"] = df["total_gen_q50"] - df["load_demand_q50"]
df["net_surplus_q10"] = df["total_gen_q10"] - df["load_demand_q10"]
df["surplus_ratio"]   = df["net_surplus_q50"] / (df["load_demand_q50"] + eps)

# Renewable fraction of load that can be covered
df["renewable_cover"] = (df["total_gen_q50"] / (df["load_demand_q50"] + eps)).clip(0, 3)

# ── 2c. Battery health merge (left join: each forecast hour gets the most
#        recent available battery daily reading) ───────────────────────────
# Round forecast timestamp to day for merging
df["date"] = df["timestamp"].dt.normalize()
df = df.sort_values("date")
batt_daily = batt_daily.sort_values("timestamp")

df = pd.merge_asof(
    df, batt_daily.rename(columns={"timestamp": "date"}),
    on="date", direction="backward"
)

# Fill any early gaps with a healthy-ish starting value
df["soh_pct"]         = df["soh_pct"].fillna(85.0)
df["deg_rate"]        = df["deg_rate"].fillna(0.0)
df["cumulative_fade"] = df["cumulative_fade"].fillna(0.0)
df["is_anomaly_any"]  = df["is_anomaly_any"].fillna(False).astype(int)

# ── 2d. Synthetic contextual features ────────────────────────────────────────
# Diesel price in Bihar (₹/kWh equivalent) — varies with global crude
# Simulate daily variation around ₹12/kWh
N = len(df)
diesel_base = 12.0
diesel_noise = np.random.normal(0, 0.5, N)
df["diesel_cost_per_kwh"] = (diesel_base + diesel_noise + 
                              1.5 * np.sin(2 * np.pi * df["month"] / 12)).clip(9, 16)

# Grid availability in rural Bihar: intermittent, better during day
# Probability of grid being up at each hour
grid_prob = np.where(df["is_daytime"] == 1, 0.55, 0.30)
df["grid_available"] = (np.random.random(N) < grid_prob).astype(int)

# Simulated State of Charge (SoC): starts around 60%, follows a daily pattern
# Lower at dawn, recharged by solar during day, drops at night
tod_soc = 50 + 25 * np.sin(np.pi * (df["hour"].values - 6) / 16)
tod_soc = np.clip(tod_soc, 20, 95)
soc_noise = np.random.normal(0, 5, N)
df["soc_pct"] = np.clip(tod_soc + soc_noise, 10, 100)

# Cell temperature rises with charging and ambient temp; peaks in summer
ambient_temp = 25 + 12 * np.sin(2 * np.pi * (df["month"].values - 3) / 12)
df["cell_temp_c"] = ambient_temp + 5 * (df["soc_pct"].values / 100) + \
                    np.random.normal(0, 2, N)
df["cell_temp_c"] = df["cell_temp_c"].clip(15, 55)

print(f"   Total feature rows : {len(df):,}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Feature column list (defined early; used in labelling + training)
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    # Forecaster outputs — quantiles
    "solar_power_q10", "solar_power_q50", "solar_power_q90",
    "wind_power_q10",  "wind_power_q50",  "wind_power_q90",
    "load_demand_q10", "load_demand_q50", "load_demand_q90",
    # Engineered supply/demand features
    "total_gen_q50", "net_surplus_q50", "net_surplus_q10",
    "surplus_ratio", "renewable_cover",
    "solar_uncertainty", "wind_uncertainty", "load_uncertainty",
    "solar_unc_score", "wind_unc_score", "load_unc_score",
    # Battery health
    "soh_pct", "soc_pct", "deg_rate", "cumulative_fade",
    "cell_temp_c", "is_anomaly_any",
    # Contextual
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "is_daytime", "is_peak_load",
    "diesel_cost_per_kwh", "grid_available",
]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Label generation (rule-based, physically motivated)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/9] Generating dispatch labels …")

# Key insight from data inspection:
#   Solar q50 range: 0 – ~12 kW (small rooftop installation)
#   Load demand q50: 22 – 102 kW (village aggregate)
# Renewables cover 5-15% of load on average — deficit is the norm.
# Dispatch rules must therefore operate on ABSOLUTE power & battery health,
# not primarily on surplus ratio.

def assign_label(row):
    """
    Returns 0 (CONSERVATIVE), 1 (MODERATE) or 2 (AGGRESSIVE).

    Bihar-calibrated dispatch rules:

    CONSERVATIVE — every situation where damage or shortage risk is high:
      - Battery critically degraded (SoH < 73%) or almost empty (SoC < 22%)
      - Battery anomaly / very high temperature (> 50 C)
      - Solar forecast extremely uncertain at night/dawn AND battery is weak
      - Diesel is expensive AND grid is unavailable AND renewables < 20% of load

    AGGRESSIVE — whenever we have a genuine renewable surplus:
      - Solar generation > 40% of load AND battery > 65% charged AND healthy
      - OR very sunny day (solar > 6 kW absolute) with good battery
      - Use the sun while it shines, charge battery, defer diesel

    MODERATE — safe default: balanced dispatch, small battery support.
    """
    soh   = row["soh_pct"]
    soc   = row["soc_pct"]
    solar = row["solar_power_q50"]    # kW
    wind  = row["wind_power_q50"]     # kW
    load  = row["load_demand_q50"]    # kW
    gen   = row["total_gen_q50"]      # kW
    temp  = row["cell_temp_c"]
    anom  = row["is_anomaly_any"]
    diesel = row["diesel_cost_per_kwh"]
    grid  = row["grid_available"]
    s_unc = row["solar_unc_score"]
    is_day = row["is_daytime"]
    cover = row["renewable_cover"]    # gen / load

    # ── CONSERVATIVE ─────────────────────────────────────────────────────────
    if soh < 73:                             return 0   # degraded battery
    if soc < 22:                             return 0   # near-empty battery
    if anom == 1:                            return 0   # fault detected
    if temp > 50:                            return 0   # thermal risk
    # Deep deficit + expensive diesel + no grid + uncertain forecast
    if (cover < 0.10) and (diesel > 13.0) and (grid == 0) and (s_unc > 3):
        return 0
    # Very low SoC + nighttime + heavy load
    if (soc < 35) and (is_day == 0) and (load > 50):
        return 0

    # ── AGGRESSIVE ───────────────────────────────────────────────────────────
    healthy = (soh >= 80) and (temp < 45)
    # Good solar output relative to load
    if (solar > 4.0) and (cover > 0.15) and (soc >= 65) and healthy:
        return 2
    # Large absolute solar surplus for the system scale
    if (solar > 7.0) and (soc >= 55) and healthy:
        return 2
    # Combined wind+solar covers > 25% of load, battery reasonably charged
    if (cover > 0.25) and (soc >= 70) and healthy:
        return 2
    # Excellent battery + cheap diesel alternative is available
    if (soc >= 80) and (soh >= 85) and (is_day == 1) and (gen > 3.0):
        return 2

    # ── MODERATE ─────────────────────────────────────────────────────────────
    return 1

df["label"] = df.apply(assign_label, axis=1)

# ── Balance classes with synthetic augmentation ───────────────────────────
# The real data skews toward CONSERVATIVE (low renewables, degraded batteries).
# We augment with synthetically varied rows to ensure all classes are learned.
count_per_class = df["label"].value_counts()
max_count = count_per_class.max()
min_target = max(int(max_count * 0.30), 1000)   # at least 30% of dominant class

synth_rows = []
for cls_idx, cls_name in enumerate(CLASS_NAMES):
    current = count_per_class.get(cls_idx, 0)
    needed  = max(0, min_target - current)
    if needed == 0:
        continue

    rng = np.random.default_rng(seed=cls_idx + 42)
    n   = needed

    if cls_idx == 0:  # CONSERVATIVE: degraded battery / night deficit
        soh_s  = rng.uniform(60, 74, n)
        soc_s  = rng.uniform(10, 30, n)
        sol_s  = rng.uniform(0.0, 1.0, n)
        win_s  = rng.uniform(0.0, 0.3, n)
        load_s = rng.uniform(30, 80, n)
        temp_s = rng.uniform(48, 55, n)
        hrs    = rng.integers(20, 24, n)          # nighttime
        months = rng.integers(1, 13, n)
        grid_s = rng.integers(0, 2, n)
        diesel_s = rng.uniform(13.0, 16.0, n)
        anom_s = rng.integers(0, 2, n)            # some anomalies

    elif cls_idx == 1:  # MODERATE: typical day, balanced
        soh_s  = rng.uniform(73, 92, n)
        soc_s  = rng.uniform(30, 68, n)
        sol_s  = rng.uniform(0.5, 4.5, n)
        win_s  = rng.uniform(0.1, 0.4, n)
        load_s = rng.uniform(35, 85, n)
        temp_s = rng.uniform(28, 46, n)
        hrs    = rng.integers(5, 22, n)
        months = rng.integers(1, 13, n)
        grid_s = rng.integers(0, 2, n)
        diesel_s = rng.uniform(10.5, 13.5, n)
        anom_s = np.zeros(n, dtype=int)

    else:  # AGGRESSIVE: sunny day, healthy battery, good SoC
        soh_s  = rng.uniform(80, 96, n)
        soc_s  = rng.uniform(65, 95, n)
        sol_s  = rng.uniform(5.0, 12.0, n)        # strong solar
        win_s  = rng.uniform(0.1, 0.8, n)
        load_s = rng.uniform(40, 80, n)
        temp_s = rng.uniform(30, 44, n)
        hrs    = rng.integers(9, 17, n)            # midday
        months = rng.integers(2, 11, n)            # spring/summer/autumn
        grid_s = rng.integers(0, 2, n)
        diesel_s = rng.uniform(9.0, 12.0, n)
        anom_s = np.zeros(n, dtype=int)

    eps = 1e-3
    gen_s = sol_s + win_s
    synth = pd.DataFrame({
        "solar_power_q10":    np.clip(sol_s * rng.uniform(0.3, 0.7, n), 0, None),
        "solar_power_q50":    sol_s,
        "solar_power_q90":    sol_s * rng.uniform(1.3, 2.5, n),
        "wind_power_q10":     np.clip(win_s * 0.4, 0, None),
        "wind_power_q50":     win_s,
        "wind_power_q90":     win_s * rng.uniform(1.5, 3.0, n),
        "load_demand_q10":    load_s * rng.uniform(0.7, 0.9, n),
        "load_demand_q50":    load_s,
        "load_demand_q90":    load_s * rng.uniform(1.1, 1.3, n),
        "total_gen_q50":      gen_s,
        "net_surplus_q50":    gen_s - load_s,
        "net_surplus_q10":    (sol_s * 0.4 + win_s * 0.4) - load_s * 1.2,
        "surplus_ratio":      (gen_s - load_s) / (load_s + eps),
        "renewable_cover":    np.clip(gen_s / (load_s + eps), 0, 3),
        "solar_uncertainty":  sol_s * rng.uniform(0.5, 2.0, n),
        "wind_uncertainty":   win_s * rng.uniform(0.8, 3.0, n),
        "load_uncertainty":   load_s * rng.uniform(0.2, 0.5, n),
        "solar_unc_score":    np.clip((sol_s * rng.uniform(0.5, 2.0, n)) / (sol_s + eps), 0, 8),
        "wind_unc_score":     np.clip((win_s * rng.uniform(0.8, 3.0, n)) / (win_s + eps), 0, 8),
        "load_unc_score":     (load_s * rng.uniform(0.2, 0.5, n)) / (load_s + eps),
        "soh_pct":            soh_s,
        "soc_pct":            soc_s,
        "deg_rate":           rng.uniform(-0.015, 0.005, n),
        "cumulative_fade":    rng.uniform(0.0, 0.35, n),
        "cell_temp_c":        temp_s,
        "is_anomaly_any":     anom_s,
        "hour_sin":           np.sin(2 * np.pi * hrs / 24),
        "hour_cos":           np.cos(2 * np.pi * hrs / 24),
        "month_sin":          np.sin(2 * np.pi * months / 12),
        "month_cos":          np.cos(2 * np.pi * months / 12),
        "is_daytime":         ((hrs >= 6) & (hrs <= 18)).astype(int),
        "is_peak_load":       (((hrs >= 7) & (hrs <= 10)) | ((hrs >= 18) & (hrs <= 22))).astype(int),
        "diesel_cost_per_kwh": diesel_s,
        "grid_available":     grid_s,
        "label":              cls_idx,
    })
    synth_rows.append(synth)
    print(f"   Synthesised {n:,} extra rows for class {cls_name}")

if synth_rows:
    df_augmented = pd.concat([df[FEATURE_COLS + ["label"]]] + synth_rows,
                              ignore_index=True)
else:
    df_augmented = df[FEATURE_COLS + ["label"]].copy()

# Add small sensor noise
for col in FEATURE_COLS:
    std = df_augmented[col].std()
    if std > 0:
        df_augmented[col] = df_augmented[col] + np.random.normal(0, std * 0.015,
                                                                    len(df_augmented))

counts = df_augmented["label"].value_counts().sort_index()
print()
for i, name in enumerate(CLASS_NAMES):
    print(f"   {name:15s}: {counts.get(i, 0):6,}  ({counts.get(i,0)/len(df_augmented)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Build feature matrix
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    # Forecaster outputs — quantiles
    "solar_power_q10", "solar_power_q50", "solar_power_q90",
    "wind_power_q10",  "wind_power_q50",  "wind_power_q90",
    "load_demand_q10", "load_demand_q50", "load_demand_q90",
    # Engineered supply/demand features
    "total_gen_q50", "net_surplus_q50", "net_surplus_q10",
    "surplus_ratio", "renewable_cover",
    "solar_uncertainty", "wind_uncertainty", "load_uncertainty",
    "solar_unc_score", "wind_unc_score", "load_unc_score",
    # Battery health
    "soh_pct", "soc_pct", "deg_rate", "cumulative_fade",
    "cell_temp_c", "is_anomaly_any",
    # Contextual
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "is_daytime", "is_peak_load",
    "diesel_cost_per_kwh", "grid_available",
]

FEATURE_LABELS = {      # Human-readable names for SHAP plots
    "solar_power_q50"   : "Solar Power (median forecast, kW)",
    "solar_power_q90"   : "Solar Power (optimistic forecast, kW)",
    "solar_power_q10"   : "Solar Power (pessimistic forecast, kW)",
    "wind_power_q50"    : "Wind Power (median forecast, kW)",
    "wind_power_q90"    : "Wind Power (optimistic, kW)",
    "wind_power_q10"    : "Wind Power (pessimistic, kW)",
    "load_demand_q50"   : "Load Demand (median forecast, kW)",
    "load_demand_q90"   : "Load Demand (high estimate, kW)",
    "load_demand_q10"   : "Load Demand (low estimate, kW)",
    "total_gen_q50"     : "Total Renewable Generation (kW)",
    "net_surplus_q50"   : "Net Energy Surplus (kW)",
    "net_surplus_q10"   : "Net Surplus – Pessimistic (kW)",
    "surplus_ratio"     : "Surplus Ratio (fraction of load)",
    "renewable_cover"   : "Renewable Cover Factor",
    "solar_uncertainty" : "Solar Forecast Uncertainty (kW)",
    "wind_uncertainty"  : "Wind Forecast Uncertainty (kW)",
    "load_uncertainty"  : "Load Forecast Uncertainty (kW)",
    "solar_unc_score"   : "Solar Uncertainty Score",
    "wind_unc_score"    : "Wind Uncertainty Score",
    "load_unc_score"    : "Load Uncertainty Score",
    "soh_pct"           : "Battery State of Health (%)",
    "soc_pct"           : "Battery State of Charge (%)",
    "deg_rate"          : "Battery Degradation Rate",
    "cumulative_fade"   : "Cumulative Capacity Fade (Ah)",
    "cell_temp_c"       : "Battery Cell Temperature (°C)",
    "is_anomaly_any"    : "Battery Anomaly Detected",
    "hour_sin"          : "Time of Day (sin)",
    "hour_cos"          : "Time of Day (cos)",
    "month_sin"         : "Season (sin)",
    "month_cos"         : "Season (cos)",
    "is_daytime"        : "Is Daytime",
    "is_peak_load"      : "Is Peak Load Period",
    "diesel_cost_per_kwh": "Diesel Cost (₹/kWh)",
    "grid_available"    : "Grid Available",
}

X = df_augmented[FEATURE_COLS].copy()
y = df_augmented["label"].copy()

# Rename columns for readability in plots
X_named = X.rename(columns=FEATURE_LABELS)
feature_display_names = [FEATURE_LABELS.get(c, c) for c in FEATURE_COLS]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Train / test split and model training
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/9] Training XGBoost classifier …")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Class weights to handle any imbalance
class_counts = y_train.value_counts()
total        = len(y_train)
class_weight = {c: total / (len(CLASS_NAMES) * class_counts[c])
                for c in class_counts.index}
sample_weights = y_train.map(class_weight).values

model = xgb.XGBClassifier(
    n_estimators          = 800,
    max_depth             = 6,
    learning_rate         = 0.05,
    subsample             = 0.85,
    colsample_bytree      = 0.80,
    min_child_weight      = 5,
    gamma                 = 0.1,
    reg_alpha             = 0.5,
    reg_lambda            = 1.5,
    eval_metric           = "mlogloss",
    early_stopping_rounds = 30,
    n_jobs                = -1,
    random_state          = 42,
    num_class             = 3,
    objective             = "multi:softprob",
    verbosity             = 0,
)

model.fit(
    X_train, y_train,
    sample_weight      = sample_weights,
    eval_set           = [(X_test, y_test)],
    verbose            = False,
)

print(f"   Best iteration: {model.best_iteration}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Evaluation
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/9] Evaluating model …")

y_pred  = model.predict(X_test).astype(int)
y_proba = model.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average="weighted")

print(f"\n   Overall Accuracy : {acc*100:.2f}%")
print(f"   Weighted F1      : {f1:.4f}")
print("\n   Classification Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
            linewidths=0.5, linecolor="white")
ax.set_title("Confusion Matrix — Dispatch Decision Engine", fontsize=13, pad=12)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("Actual",    fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()
print("   → Saved confusion_matrix.png")

# Save model
model.save_model(os.path.join(OUT_DIR, "dispatch_engine.ubj"))
print("   → Saved dispatch_engine.ubj")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — SHAP global explanations
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/9] Computing SHAP values (global explanations) …")

explainer   = shap.TreeExplainer(model)
shap_values = explainer(X_test)    # shape: (n_samples, n_features, n_classes)

# ── 7a. Beeswarm summary — one per class ─────────────────────────────────────
for cls_idx, cls_name in enumerate(CLASS_NAMES):
    fig, ax = plt.subplots(figsize=(10, 9))
    plt.sca(ax)
    shap.summary_plot(
        shap_values[:, :, cls_idx].values,
        X_test,
        feature_names   = feature_display_names,
        show            = False,
        max_display     = 20,
        plot_type       = "dot",
        color_bar_label = "Feature value",
    )
    plt.title(f"SHAP Beeswarm — Class: {cls_name}", fontsize=13, pad=10)
    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f"shap_beeswarm_{cls_name.lower()}.png")
    plt.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"   → Saved shap_beeswarm_{cls_name.lower()}.png")

# ── 7b. Feature importance bar chart (mean |SHAP| across all classes) ────────
mean_abs_shap = np.abs(shap_values.values).mean(axis=(0, 2))   # (n_features,)
importance_df = pd.DataFrame({
    "feature"      : feature_display_names,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=True).tail(20)

fig, ax = plt.subplots(figsize=(9, 8))
bars = ax.barh(importance_df["feature"], importance_df["mean_abs_shap"],
               color="#4e79a7", edgecolor="white", height=0.7)
ax.set_xlabel("Mean |SHAP value| (impact on model output)", fontsize=11)
ax.set_title("Feature Importance — Dispatch Decision Engine\n(averaged across all dispatch classes)",
             fontsize=12, pad=10)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap_feature_importance.png"), dpi=150, bbox_inches="tight")
plt.close()
print("   → Saved shap_feature_importance.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Per-sample plain-English explanations
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/9] Generating per-sample plain-English explanations …")

# Readable descriptions for each feature in operator language
OPERATOR_DESC = {
    "solar_power_q50"    : "solar generation (best estimate)",
    "solar_power_q90"    : "solar generation (optimistic estimate)",
    "solar_power_q10"    : "solar generation (pessimistic estimate)",
    "wind_power_q50"     : "wind power output",
    "wind_power_q90"     : "wind power (optimistic estimate)",
    "wind_power_q10"     : "wind power (pessimistic estimate)",
    "load_demand_q50"    : "village power demand",
    "load_demand_q90"    : "village demand (high estimate)",
    "load_demand_q10"    : "village demand (low estimate)",
    "total_gen_q50"      : "total solar + wind generation",
    "net_surplus_q50"    : "surplus energy after meeting demand",
    "net_surplus_q10"    : "surplus in the worst-case scenario",
    "surplus_ratio"      : "how much extra energy is available",
    "renewable_cover"    : "how well renewables cover the load",
    "solar_uncertainty"  : "uncertainty in solar forecast",
    "wind_uncertainty"   : "uncertainty in wind forecast",
    "load_uncertainty"   : "uncertainty in demand forecast",
    "solar_unc_score"    : "solar forecast reliability",
    "wind_unc_score"     : "wind forecast reliability",
    "load_unc_score"     : "demand forecast reliability",
    "soh_pct"            : "battery health level",
    "soc_pct"            : "battery charge level",
    "deg_rate"           : "battery degradation speed",
    "cumulative_fade"    : "total battery wear so far",
    "cell_temp_c"        : "battery cell temperature",
    "is_anomaly_any"     : "battery fault/anomaly status",
    "hour_sin"           : "time of day",
    "hour_cos"           : "time of day",
    "month_sin"          : "season of the year",
    "month_cos"          : "season of the year",
    "is_daytime"         : "daytime/nighttime indicator",
    "is_peak_load"       : "peak demand period",
    "diesel_cost_per_kwh": "current diesel price",
    "grid_available"     : "grid power availability",
}

RECOMMENDATIONS = {
    0: "Limit battery discharge. Rely on diesel or grid if available. "
       "Allow solar to slow-charge the battery without heavy loads.",
    1: "Balance renewable supply with battery support. "
       "Charge battery when surplus, draw modestly during deficit.",
    2: "Maximise battery charging and renewable usage. "
       "If surplus is large, consider deferring heavy loads to this window.",
}


def shap_direction(val):
    """Returns 'pushed toward' or 'pushed away from' based on SHAP sign."""
    return "pushed toward" if val > 0 else "pushed away from"


def generate_plain_english(idx, X_row, shap_row, pred_class, confidence, row_data):
    """
    Build a human-readable explanation for one sample.

    Parameters
    ----------
    idx        : int   — row index in X_test
    X_row      : Series — feature values for this sample
    shap_row   : array  — SHAP values for the predicted class (n_features,)
    pred_class : int   — predicted class index
    confidence : float — probability for pred_class
    row_data   : Series — full row from df for contextual printing

    Returns
    -------
    str : plain-English explanation
    """
    class_name = CLASS_NAMES[pred_class]

    # Top 3 features by |SHAP| for the predicted class
    top_indices = np.argsort(np.abs(shap_row))[-3:][::-1]

    driver_lines = []
    for rank, fi in enumerate(top_indices, 1):
        feat          = FEATURE_COLS[fi]
        feat_readable = OPERATOR_DESC.get(feat, feat)
        shap_val      = shap_row[fi]
        feat_val      = X_row.iloc[fi]

        # Format value nicely
        if feat in ["is_daytime", "is_peak_load", "grid_available", "is_anomaly_any"]:
            val_str = "YES" if feat_val > 0.5 else "NO"
        elif feat in ["hour_sin", "hour_cos", "month_sin", "month_cos"]:
            val_str = f"{feat_val:.2f}"
        else:
            val_str = f"{feat_val:.2f}"

        direction = "increased" if shap_val > 0 else "decreased"
        driver_lines.append(
            f"  {rank}. {feat_readable.capitalize()} "
            f"(value: {val_str}) — {direction} likelihood of {class_name}"
        )

    explanation = (
        f"DECISION: {class_name}  (confidence: {confidence*100:.1f}%)\n"
        f"WHY THIS DECISION WAS MADE:\n"
        + "\n".join(driver_lines) + "\n"
        f"RECOMMENDATION: {RECOMMENDATIONS[pred_class]}"
    )
    return explanation


# Generate for all test samples — store in DataFrame
test_indices = X_test.index.tolist()
explanations = []

for i, (idx, x_row) in enumerate(X_test.iterrows()):
    pred     = int(y_pred[i])
    conf     = float(y_proba[i][pred])
    sv_pred  = shap_values[i, :, pred].values    # SHAP for predicted class

    exp_str = generate_plain_english(
        idx      = i,
        X_row    = x_row,
        shap_row = sv_pred,
        pred_class = pred,
        confidence = conf,
        row_data   = df.loc[idx] if idx in df.index else pd.Series(),
    )
    explanations.append({
        "idx"         : idx,
        "true_label"  : CLASS_NAMES[int(y_test.iloc[i])],
        "pred_label"  : CLASS_NAMES[pred],
        "confidence"  : round(conf, 4),
        "explanation" : exp_str,
    })

exp_df = pd.DataFrame(explanations)
exp_df.to_csv(os.path.join(OUT_DIR, "per_sample_explanations.csv"),
              index=False)
print(f"   → Saved per_sample_explanations.csv ({len(exp_df):,} explanations)")

# Print first 3 to console as preview
print("\n   ── Sample explanations preview ─────────────────────────────────")
for row in explanations[:3]:
    print(f"\n   [Sample idx={row['idx']}]")
    for line in row["explanation"].split("\n"):
        print(f"     {line}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Waterfall + Force plots for one sample from each class
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8/9] Generating waterfall and force plots …")

# Map test indices back to positions in X_test
X_test_reset    = X_test.reset_index(drop=True)
y_test_reset    = y_test.reset_index(drop=True)
y_pred_arr      = np.array(y_pred)

for cls_idx, cls_name in enumerate(CLASS_NAMES):
    # Find first correctly classified sample for this class
    mask = (y_test_reset == cls_idx) & (y_pred_arr == cls_idx)
    candidates = np.where(mask)[0]
    if len(candidates) == 0:
        # Fallback: first predicted as this class
        candidates = np.where(y_pred_arr == cls_idx)[0]
    sample_pos = candidates[0]

    # ── Waterfall plot ──────────────────────────────────────────────────────
    sv_sample = shap_values[sample_pos, :, cls_idx]
    sv_sample.feature_names = feature_display_names

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.sca(ax)
    shap.plots.waterfall(sv_sample, max_display=15, show=False)
    plt.title(f"SHAP Waterfall — {cls_name}\n"
              f"(Confidence: {y_proba[sample_pos][cls_idx]*100:.1f}%)",
              fontsize=12, pad=10)
    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f"shap_waterfall_{cls_name.lower()}.png")
    plt.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"   → Saved shap_waterfall_{cls_name.lower()}.png")

    # ── Force plot (saved as HTML) ─────────────────────────────────────────
    fp = shap.force_plot(
        explainer.expected_value[cls_idx],
        shap_values[sample_pos, :, cls_idx].values,
        X_test_reset.iloc[sample_pos],
        feature_names = feature_display_names,
    )
    force_html = shap.save_html.__module__    # just checking shap version
    html_path  = os.path.join(OUT_DIR, f"shap_force_{cls_name.lower()}.html")
    shap.save_html(html_path, fp)
    print(f"   → Saved shap_force_{cls_name.lower()}.html")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — predict_and_explain()
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9/9] Defining predict_and_explain() and running live demo …")


def predict_and_explain(
    solar_q10, solar_q50, solar_q90,
    wind_q10,  wind_q50,  wind_q90,
    load_q10,  load_q50,  load_q90,
    soh_pct, soc_pct, cell_temp_c,
    hour, month,
    diesel_cost=12.0, grid_available=0,
    deg_rate=0.0, cumulative_fade=0.0, is_anomaly=0,
    _model=model, _explainer=explainer, _feature_names=feature_display_names,
    _feature_cols=FEATURE_COLS,
):
    """
    Live inference function for the dispatch decision engine.

    Parameters
    ----------
    solar_q10/q50/q90   : float — quantile solar forecast (kW)
    wind_q10/q50/q90    : float — quantile wind forecast (kW)
    load_q10/q50/q90    : float — quantile load demand (kW)
    soh_pct             : float — battery state of health (%)
    soc_pct             : float — battery state of charge (%)
    cell_temp_c         : float — battery cell temperature (°C)
    hour                : int   — hour of day (0–23)
    month               : int   — month (1–12)
    diesel_cost         : float — diesel cost ₹/kWh (default 12)
    grid_available      : int   — 1 if grid is up, else 0
    deg_rate            : float — battery degradation rate (from model 2)
    cumulative_fade     : float — total fade in Ah (from model 2)
    is_anomaly          : int   — 1 if battery anomaly detected, else 0

    Returns
    -------
    dict with keys:
        decision        : str   — "CONSERVATIVE" / "MODERATE" / "AGGRESSIVE"
        confidence      : float — probability (0–1)
        class_probs     : dict  — probabilities for all three classes
        explanation     : str   — plain-English explanation
        shap_values     : dict  — {feature_name: shap_value}
        recommendation  : str   — one-line operator action
    """
    eps = 1e-3

    # Build feature vector matching training columns
    solar_q10 = max(solar_q10, 0);  solar_q50 = max(solar_q50, 0);  solar_q90 = max(solar_q90, 0)
    wind_q10  = max(wind_q10,  0);  wind_q50  = max(wind_q50,  0);  wind_q90  = max(wind_q90,  0)

    total_gen  = solar_q50 + wind_q50
    total_q10  = solar_q10 + wind_q10
    surplus    = total_gen - load_q50
    surp_q10   = total_q10 - load_q90   # pessimistic surplus

    row = {
        "solar_power_q10"   : solar_q10,
        "solar_power_q50"   : solar_q50,
        "solar_power_q90"   : solar_q90,
        "wind_power_q10"    : wind_q10,
        "wind_power_q50"    : wind_q50,
        "wind_power_q90"    : wind_q90,
        "load_demand_q10"   : load_q10,
        "load_demand_q50"   : load_q50,
        "load_demand_q90"   : load_q90,
        "total_gen_q50"     : total_gen,
        "net_surplus_q50"   : surplus,
        "net_surplus_q10"   : surp_q10,
        "surplus_ratio"     : surplus / (load_q50 + eps),
        "renewable_cover"   : min(total_gen / (load_q50 + eps), 3),
        "solar_uncertainty" : solar_q90 - solar_q10,
        "wind_uncertainty"  : wind_q90  - wind_q10,
        "load_uncertainty"  : load_q90  - load_q10,
        "solar_unc_score"   : (solar_q90 - solar_q10) / (solar_q50 + eps),
        "wind_unc_score"    : (wind_q90  - wind_q10)  / (wind_q50  + eps),
        "load_unc_score"    : (load_q90  - load_q10)  / (load_q50  + eps),
        "soh_pct"           : soh_pct,
        "soc_pct"           : soc_pct,
        "deg_rate"          : deg_rate,
        "cumulative_fade"   : cumulative_fade,
        "cell_temp_c"       : cell_temp_c,
        "is_anomaly_any"    : is_anomaly,
        "hour_sin"          : np.sin(2 * np.pi * hour  / 24),
        "hour_cos"          : np.cos(2 * np.pi * hour  / 24),
        "month_sin"         : np.sin(2 * np.pi * month / 12),
        "month_cos"         : np.cos(2 * np.pi * month / 12),
        "is_daytime"        : int(6 <= hour <= 18),
        "is_peak_load"      : int((7 <= hour <= 10) or (18 <= hour <= 22)),
        "diesel_cost_per_kwh": diesel_cost,
        "grid_available"    : grid_available,
    }

    X_live   = pd.DataFrame([row])[_feature_cols]
    proba    = _model.predict_proba(X_live)[0]
    pred_cls = int(np.argmax(proba))
    conf     = float(proba[pred_cls])

    # SHAP for live sample
    sv       = _explainer(X_live)
    sv_pred  = sv[0, :, pred_cls].values

    # Top 3 drivers
    top3 = np.argsort(np.abs(sv_pred))[-3:][::-1]
    driver_lines = []
    for rank, fi in enumerate(top3, 1):
        feat   = _feature_cols[fi]
        rdbl   = OPERATOR_DESC.get(feat, feat)
        val    = X_live.iloc[0, fi]
        direct = "increased" if sv_pred[fi] > 0 else "decreased"
        driver_lines.append(
            f"  {rank}. {rdbl.capitalize()} "
            f"(= {val:.2f}) — {direct} likelihood of {CLASS_NAMES[pred_cls]}"
        )

    explanation = (
        f"DECISION: {CLASS_NAMES[pred_cls]}  (confidence: {conf*100:.1f}%)\n"
        f"WHY:\n" + "\n".join(driver_lines) + "\n"
        f"RECOMMENDATION: {RECOMMENDATIONS[pred_cls]}"
    )

    shap_dict = {
        _feature_names[i]: float(sv_pred[i])
        for i in range(len(_feature_cols))
    }

    return {
        "decision"       : CLASS_NAMES[pred_cls],
        "confidence"     : conf,
        "class_probs"    : {CLASS_NAMES[i]: float(proba[i]) for i in range(3)},
        "explanation"    : explanation,
        "shap_values"    : shap_dict,
        "recommendation" : RECOMMENDATIONS[pred_cls],
    }


# ── Live demo calls ──────────────────────────────────────────────────────────
demo_cases = [
    dict(
        desc        = "Morning peak (07:00): good solar, healthy battery",
        solar_q10=2.0, solar_q50=8.5,  solar_q90=14.0,
        wind_q10=0.5,  wind_q50=1.2,   wind_q90=2.5,
        load_q10=45.0, load_q50=60.0,  load_q90=75.0,
        soh_pct=88.0, soc_pct=72.0, cell_temp_c=36.0,
        hour=7, month=4, diesel_cost=12.5, grid_available=0,
    ),
    dict(
        desc        = "Noon (12:00): peak solar, high SoC, large surplus",
        solar_q10=9.0,  solar_q50=18.0, solar_q90=24.0,
        wind_q10=1.0,   wind_q50=2.5,   wind_q90=4.5,
        load_q10=55.0,  load_q50=68.0,  load_q90=82.0,
        soh_pct=91.0, soc_pct=82.0, cell_temp_c=42.0,
        hour=12, month=6, diesel_cost=11.0, grid_available=0,
    ),
    dict(
        desc        = "Night (22:00): no solar, degraded battery, grid down",
        solar_q10=0.0, solar_q50=0.0,  solar_q90=0.0,
        wind_q10=0.2,  wind_q50=0.8,   wind_q90=1.8,
        load_q10=30.0, load_q50=42.0,  load_q90=55.0,
        soh_pct=68.0, soc_pct=18.0, cell_temp_c=52.0,
        hour=22, month=1, diesel_cost=14.0, grid_available=0,
        is_anomaly=1,
    ),
]

print("\n" + "─" * 72)
print("  LIVE INFERENCE DEMO — predict_and_explain()")
print("─" * 72)
for i, case in enumerate(demo_cases, 1):
    d = case.pop("desc")
    result = predict_and_explain(**case)
    print(f"\n  Demo {i}: {d}")
    print(f"  {'─'*66}")
    for line in result["explanation"].split("\n"):
        print(f"  {line}")
    probs = result["class_probs"]
    print(f"  Class probabilities: "
          f"CON={probs['CONSERVATIVE']:.3f}  "
          f"MOD={probs['MODERATE']:.3f}  "
          f"AGG={probs['AGGRESSIVE']:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  ALL OUTPUTS SAVED TO:", OUT_DIR)
print("=" * 72)
outputs = [
    "confusion_matrix.png",
    "dispatch_engine.ubj",
    "shap_beeswarm_conservative.png",
    "shap_beeswarm_moderate.png",
    "shap_beeswarm_aggressive.png",
    "shap_feature_importance.png",
    "per_sample_explanations.csv",
    "shap_waterfall_conservative.png",
    "shap_waterfall_moderate.png",
    "shap_waterfall_aggressive.png",
    "shap_force_conservative.html",
    "shap_force_moderate.html",
    "shap_force_aggressive.html",
]
for f in outputs:
    path = os.path.join(OUT_DIR, f)
    exists = "✔" if os.path.exists(path) else "✘"
    print(f"  {exists}  {f}")
print()
