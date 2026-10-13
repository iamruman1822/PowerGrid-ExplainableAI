"""
04_generate_2yr_inference.py
============================
Generates 2 years of realistic synthetic battery cycle data (730 cycles @ 1/day)
then runs the trained LSTM to produce full inference outputs:
  - Predicted capacity vs synthetic true
  - State of Health (SoH %)
  - Remaining Useful Life (RUL) in days
  - Degradation rate per cycle
  - Anomaly detection
  - 30-day future forecast beyond 2 years

INPUT SHAPE TO MODEL: (batch, 10, 7)
Features per cycle (in exact order the scaler expects):
  [0] mean_voltage        (V)
  [1] min_voltage         (V)
  [2] mean_current        (A)
  [3] mean_temp           (°C)
  [4] max_temp            (°C)
  [5] discharge_duration  (s)
  [6] cycle_num_norm      (0.0 – 1.0)
"""

import os, sys, pickle, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from datetime import date, timedelta
import tensorflow as tf
from tensorflow import keras
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(DATA_DIR, 'outputs')
PLOT_DIR    = os.path.join(OUTPUT_DIR, 'plots')
DATA_OUT    = os.path.join(OUTPUT_DIR, 'data')
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(DATA_OUT, exist_ok=True)

SEQ_LEN       = 10            # must match trained model
TOTAL_CYCLES  = 730           # 2 years × 365 days/year
FORECAST_DAYS = 30            # extra forecast beyond 2 years
EOL_CAP       = 1.6           # Ahr — 80% of 2.0 Ahr nominal
NOMINAL_CAP   = 2.0
ANOMALY_Z     = 2.0
SEED          = 42
np.random.seed(SEED)

# ─────────────────────────────────────────────
# LOAD MODEL + SCALERS
# ─────────────────────────────────────────────
def require(path, name):
    if not os.path.exists(path):
        sys.exit(f"ERROR: {name} not found – run earlier scripts first.")
    return path

model    = keras.models.load_model(require(os.path.join(DATA_DIR, 'lstm_battery_model.keras'), 'LSTM model'))
scaler_X = pickle.load(open(require(os.path.join(DATA_OUT, 'scaler_X.pkl'), 'scaler_X'), 'rb'))
scaler_y = pickle.load(open(require(os.path.join(DATA_OUT, 'scaler_y.pkl'), 'scaler_y'), 'rb'))
meta     = json.load(open(os.path.join(DATA_OUT, 'model_metadata.json')))

print("=" * 65)
print("  2-YEAR BATTERY DEGRADATION SIMULATION + LSTM INFERENCE")
print("=" * 65)
print(f"\nModel input:  (batch, {SEQ_LEN}, 7) — 7 features per cycle")
print(f"Simulating {TOTAL_CYCLES} cycles ({TOTAL_CYCLES/365:.1f} years @ 1 cycle/day)\n")

FEATURE_COLS = meta['feature_cols']   # correct order
print(f"Feature order for inference:")
for i, f in enumerate(FEATURE_COLS):
    print(f"  [{i}] {f}")
print()

# ─────────────────────────────────────────────
# SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────
# Physical degradation model:
#   capacity(c) = C0 - alpha * c - beta * ln(c+1) + noise
#   voltage sags gently,  temp rises slightly,  duration shortens
# Calibrated to match NASA battery behavior (B0025-class)

def generate_2yr_battery_data(
    total_cycles    = 730,
    C0              = 1.95,     # initial capacity (slightly below nominal due to formation)
    alpha           = 0.00060,  # linear fade per cycle
    beta            = 0.00018,  # logarithmic fade coefficient
    noise_std       = 0.005,    # cycle-to-cycle capacity noise
    v_init          = 3.72,     # initial mean voltage
    v_fade          = 0.00015,  # voltage fade per cycle
    i_mean          = -1.55,    # mean discharge current (negative = discharge)
    temp_init       = 25.0,     # initial temperature
    temp_rise       = 0.003,    # temperature rise per cycle (thermal aging)
    dur_init        = 3600.0,   # initial discharge duration (s)
    dur_fade        = 2.2,      # seconds shorter per cycle
    seed            = 42,
):
    rng = np.random.default_rng(seed)

    cycles   = np.arange(1, total_cycles + 1)
    dates    = [date(2024, 1, 1) + timedelta(days=int(c - 1)) for c in cycles]

    # Capacity — physical degradation
    cap_true = (
        C0
        - alpha * cycles
        - beta  * np.log(cycles + 1)
        + rng.normal(0, noise_std, total_cycles)
    )
    cap_true = np.clip(cap_true, 0.3, NOMINAL_CAP)   # physical min

    # Voltage — sags with age, small random fluctuation
    mean_voltage = v_init - v_fade * cycles + rng.normal(0, 0.005, total_cycles)
    min_voltage  = mean_voltage - (0.35 + 0.00008 * cycles) + rng.normal(0, 0.008, total_cycles)
    min_voltage  = np.clip(min_voltage, 2.0, 4.2)

    # Current — roughly constant with small noise
    mean_current = np.full(total_cycles, i_mean) + rng.normal(0, 0.02, total_cycles)

    # Temperature — rises slowly (more internal resistance → more heat)
    mean_temp = temp_init + temp_rise * cycles + rng.normal(0, 0.3, total_cycles)
    max_temp  = mean_temp + 2.5 + 0.002 * cycles + rng.normal(0, 0.2, total_cycles)

    # Discharge duration — proportional to capacity
    dur = dur_init - dur_fade * cycles + rng.normal(0, 30, total_cycles)
    dur = np.clip(dur, 300, dur_init * 1.05)

    # Normalised cycle number (0 → 1 over the lifespan)
    cycle_num_norm = (cycles - 1) / (total_cycles - 1)

    df = pd.DataFrame({
        'date'             : dates,
        'cycle_num'        : cycles,
        'capacity_true'    : cap_true,
        'mean_voltage'     : mean_voltage,
        'min_voltage'      : min_voltage,
        'mean_current'     : mean_current,
        'mean_temp'        : mean_temp,
        'max_temp'         : max_temp,
        'discharge_duration': dur,
        'cycle_num_norm'   : cycle_num_norm,
    })
    return df


df_syn = generate_2yr_battery_data(TOTAL_CYCLES)
print(f"Synthetic dataset generated: {len(df_syn)} cycles")
print(f"  Date range : {df_syn['date'].iloc[0]} → {df_syn['date'].iloc[-1]}")
print(f"  Cap range  : {df_syn['capacity_true'].min():.4f} – {df_syn['capacity_true'].max():.4f} Ahr")
print()


# ─────────────────────────────────────────────
# SCALE + BUILD SEQUENCES
# ─────────────────────────────────────────────
X_raw = df_syn[FEATURE_COLS].values       # (730, 7)
y_raw = df_syn[['capacity_true']].values  # (730, 1)

X_sc = scaler_X.transform(X_raw)
y_sc = scaler_y.transform(y_raw)

# Sliding window
Xs, ys, pred_cycles, pred_dates = [], [], [], []
for i in range(SEQ_LEN, len(X_sc)):
    Xs.append(X_sc[i - SEQ_LEN : i])
    ys.append(y_sc[i, 0])
    pred_cycles.append(df_syn['cycle_num'].iloc[i])
    pred_dates.append(df_syn['date'].iloc[i])

Xs = np.array(Xs, dtype=np.float32)   # (720, 10, 7)
ys = np.array(ys, dtype=np.float32)

# ─────────────────────────────────────────────
# INFERENCE — model.predict
# ─────────────────────────────────────────────
print(f"Running LSTM inference on {len(Xs)} sequences (shape={Xs.shape}) ...")
pred_sc = model.predict(Xs, verbose=0).flatten()
pred_cap = scaler_y.inverse_transform(pred_sc.reshape(-1, 1)).flatten()
true_cap = scaler_y.inverse_transform(ys.reshape(-1, 1)).flatten()

# Assemble results DataFrame
results = pd.DataFrame({
    'date'         : pred_dates,
    'cycle_num'    : pred_cycles,
    'capacity_true': true_cap,
    'capacity_pred': pred_cap,
})

# ─────────────────────────────────────────────
# COMPUTE DERIVED METRICS
# ─────────────────────────────────────────────
# State of Health
cap_init_true = true_cap[0]
results['soh_true_%']    = (results['capacity_true'] / NOMINAL_CAP) * 100
results['soh_pred_%']    = (results['capacity_pred'] / NOMINAL_CAP) * 100

# Degradation Rate
results['deg_rate_true'] = results['capacity_true'].diff() / results['cycle_num'].diff()
results['deg_rate_pred'] = results['capacity_pred'].diff() / results['cycle_num'].diff()
results['deg_rate_true_smooth'] = results['deg_rate_true'].rolling(7, center=True).mean()
results['deg_rate_pred_smooth'] = results['deg_rate_pred'].rolling(7, center=True).mean()

# Anomaly Detection
results['pred_error'] = results['capacity_pred'] - results['capacity_true']
err_mean = results['pred_error'].mean()
err_std  = results['pred_error'].std() + 1e-8
results['error_z']    = (results['pred_error'] - err_mean) / err_std
results['is_anomaly'] = results['error_z'].abs() > ANOMALY_Z

# RUL (cycles to EOL at current trend)
def compute_rul(cap_series, cycle_series, eol_cap):
    """Estimate remaining cycles from the last point."""
    below = cap_series[cap_series < eol_cap]
    if not below.empty:
        eol_cycle = cycle_series[below.index[0]]
        return max(0, int(eol_cycle - cycle_series.iloc[-1]))
    # Extrapolate via linear fit on last 30 cycles
    last_n = min(30, len(cap_series))
    x = cycle_series.tail(last_n).values
    y = cap_series.tail(last_n).values
    if len(x) < 2: return np.nan
    slope = np.polyfit(x, y, 1)[0]
    if slope >= 0: return 9999   # not degrading
    cur_cap = cap_series.iloc[-1]
    return max(0, int((cur_cap - eol_cap) / abs(slope)))

rul_true = compute_rul(results['capacity_true'], results['cycle_num'], EOL_CAP)
rul_pred = compute_rul(results['capacity_pred'], results['cycle_num'], EOL_CAP)
rul_days_true = rul_true  # 1 cycle/day
rul_days_pred = rul_pred

# MAE / RMSE
mae  = np.mean(np.abs(pred_cap - true_cap))
rmse = np.sqrt(np.mean((pred_cap - true_cap)**2))
mape = np.mean(np.abs((pred_cap - true_cap) / (true_cap + 1e-8))) * 100

print(f"\n── 2-Year Inference Metrics ────────────────────────────")
print(f"  MAE  = {mae:.4f} Ahr")
print(f"  RMSE = {rmse:.4f} Ahr")
print(f"  MAPE = {mape:.2f} %")
print(f"  Avg Degradation Rate = {results['deg_rate_true'].dropna().mean()*1000:.4f} mAhr/cycle")
print(f"  Final SoH (true)    = {results['soh_true_%'].iloc[-1]:.2f} %")
print(f"  Final SoH (pred)    = {results['soh_pred_%'].iloc[-1]:.2f} %")
print(f"  RUL (true)          = {rul_true} cycles = {rul_days_true} days")
print(f"  RUL (predicted)     = {rul_pred} cycles = {rul_days_pred} days")
print(f"  Anomalies detected  = {results['is_anomaly'].sum()} cycles")


# ─────────────────────────────────────────────
# AUTOREGRESSIVE FORECAST (30 days beyond 2 yrs)
# ─────────────────────────────────────────────
def forecast_ar(seed_seq_sc, steps, last_cycle_norm, total):
    """Extend forecast auto-regressively."""
    seq = seed_seq_sc.copy()
    preds, norm_vals = [], []
    for s in range(steps):
        inp = seq.reshape(1, SEQ_LEN, -1)
        p = model.predict(inp, verbose=0)[0, 0]
        p_cap = float(scaler_y.inverse_transform([[p]])[0, 0])
        preds.append(p_cap)
        new_row = seq[-1].copy()
        new_norm = min(1.0, last_cycle_norm + (s + 1) / total)
        new_row[FEATURE_COLS.index('cycle_num_norm')] = new_norm
        seq = np.vstack([seq[1:], new_row])
        norm_vals.append(new_norm)
    return preds

last_seq = X_sc[-SEQ_LEN:].copy()
last_cycle_norm = df_syn['cycle_num_norm'].iloc[-1]
fut_caps = forecast_ar(last_seq, FORECAST_DAYS, last_cycle_norm, TOTAL_CYCLES + FORECAST_DAYS)
fut_dates = [df_syn['date'].iloc[-1] + timedelta(days=i+1) for i in range(FORECAST_DAYS)]
fut_cycles = [TOTAL_CYCLES + i + 1 for i in range(FORECAST_DAYS)]

# ─────────────────────────────────────────────
# SAVE CSV
# ─────────────────────────────────────────────
results.to_csv(os.path.join(DATA_OUT, '2yr_simulation_results.csv'), index=False)
df_syn.to_csv(os.path.join(DATA_OUT, '2yr_synthetic_raw_data.csv'), index=False)

forecast_df = pd.DataFrame({
    'date': fut_dates, 'cycle_num': fut_cycles,
    'forecast_capacity_Ahr': fut_caps,
    'forecast_soh_%': [c / NOMINAL_CAP * 100 for c in fut_caps],
})
forecast_df.to_csv(os.path.join(DATA_OUT, '2yr_30day_forecast.csv'), index=False)

print(f"\n[CSV] 2yr_simulation_results.csv")
print(f"[CSV] 2yr_synthetic_raw_data.csv")
print(f"[CSV] 2yr_30day_forecast.csv")


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────
dates_dt = pd.to_datetime(results['date'])
fut_dates_dt = pd.to_datetime(fut_dates)


# ── PLOT A: Master dashboard (2×3 grid) ──────
print("\n[PLOT] A — 2-Year Master Dashboard ...")
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor('#0f0f1a')
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

STYLE = dict(facecolor='#0f0f1a', fc='#0f0f1a')
TEXT  = dict(color='white')
GRID  = dict(alpha=0.15, color='white')

def dark_ax(ax):
    ax.set_facecolor('#1a1a2e')
    for spine in ax.spines.values():
        spine.set_color('#444466')
    ax.tick_params(colors='white', labelsize=8)
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.grid(**GRID)
    return ax

# ── Panel 1: Capacity Fade ──
ax1 = dark_ax(fig.add_subplot(gs[0, :2]))
ax1.plot(dates_dt, results['capacity_true'], color='#4fc3f7', linewidth=2,   label='Synthetic True', alpha=0.9)
ax1.plot(dates_dt, results['capacity_pred'], color='#ff7043', linewidth=1.8, label='LSTM Predicted', linestyle='--', alpha=0.9)
ax1.fill_between(dates_dt,
    results['capacity_true'] - 0.02, results['capacity_true'] + 0.02,
    alpha=0.1, color='#4fc3f7')
# Forecast
ax1.plot(fut_dates_dt, fut_caps, color='#a5d6a7', linewidth=2, linestyle=':',
         marker='D', markersize=3, label=f'+{FORECAST_DAYS}d Forecast')
ax1.axhline(EOL_CAP, color='#ef5350', linestyle='-.', linewidth=1.5, label='EOL (1.6 Ahr)')
ax1.set_title('Capacity Degradation — 2 Years + 30-Day Forecast', fontweight='bold', fontsize=11)
ax1.set_ylabel('Capacity (Ahr)'); ax1.legend(fontsize=7, loc='upper right')
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')

# ── Panel 2: SoH ──
ax2 = dark_ax(fig.add_subplot(gs[0, 2]))
ax2.plot(dates_dt, results['soh_true_%'], color='#4fc3f7', linewidth=1.8, label='True SoH')
ax2.plot(dates_dt, results['soh_pred_%'], color='#ff7043', linewidth=1.5, linestyle='--', label='Pred SoH')
ax2.axhline(80, color='#ef5350', linestyle='-.', linewidth=1.5, label='EOL 80%')
ax2.set_ylim([0, 115]); ax2.set_title('State of Health (%)', fontweight='bold', fontsize=10)
ax2.set_ylabel('SoH (%)'); ax2.legend(fontsize=7)

# ── Panel 3: Degradation Rate ──
ax3 = dark_ax(fig.add_subplot(gs[1, :2]))
ax3.plot(dates_dt, results['deg_rate_true_smooth'] * 1000, color='#80cbc4', linewidth=1.8, label='Actual (mAhr/cycle)')
ax3.plot(dates_dt, results['deg_rate_pred_smooth'] * 1000, color='#ffb74d', linewidth=1.5, linestyle='--', label='Predicted')
ax3.axhline(0, color='white', linewidth=0.6, linestyle=':')
ax3.set_title('Degradation Rate (ΔAhr/cycle × 1000), 7-cycle smooth', fontweight='bold', fontsize=10)
ax3.set_ylabel('mAhr/cycle'); ax3.legend(fontsize=7)
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha='right')

# ── Panel 4: Error ──
ax4 = dark_ax(fig.add_subplot(gs[1, 2]))
anomaly_dates = dates_dt[results['is_anomaly']]
normal_dates  = dates_dt[~results['is_anomaly']]
ax4.plot(normal_dates, results.loc[~results['is_anomaly'], 'pred_error'],
         '.', color='#90caf9', markersize=3, label='Normal')
ax4.scatter(anomaly_dates, results.loc[results['is_anomaly'], 'pred_error'],
            c='#ef5350', s=30, zorder=5, label=f'Anomaly ({results["is_anomaly"].sum()})')
ax4.axhline(0, color='white', linewidth=0.7, linestyle='--')
ax4.set_title('Prediction Error + Anomalies', fontweight='bold', fontsize=10); ax4.set_ylabel('Error (Ahr)')
ax4.legend(fontsize=7)

# ── Panel 5: Cumulative Degradation ──
ax5 = dark_ax(fig.add_subplot(gs[2, :2]))
cum_fade_true = results['capacity_true'].iloc[0] - results['capacity_true']
cum_fade_pred = results['capacity_pred'].iloc[0] - results['capacity_pred']
ax5.fill_between(dates_dt, cum_fade_true, alpha=0.35, color='#4fc3f7', label='True fade')
ax5.plot(dates_dt, cum_fade_true, color='#4fc3f7', linewidth=1.8)
ax5.plot(dates_dt, cum_fade_pred, color='#ff7043', linewidth=1.5, linestyle='--', label='Predicted fade')
ax5.set_title('Cumulative Capacity Fade from Initial', fontweight='bold', fontsize=10)
ax5.set_ylabel('Capacity Lost (Ahr)'); ax5.legend(fontsize=7)
ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax5.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=30, ha='right')

# ── Panel 6: KPI Box ──
ax6 = dark_ax(fig.add_subplot(gs[2, 2]))
ax6.axis('off')
kpi_text = (
    f"  2-YEAR SUMMARY\n"
    f"{'─'*26}\n"
    f"  Cycles simulated : 730\n"
    f"  Duration         : 2 years\n\n"
    f"  ── Accuracy ─────────────\n"
    f"  MAE   : {mae:.4f} Ahr\n"
    f"  RMSE  : {rmse:.4f} Ahr\n"
    f"  MAPE  : {mape:.2f} %\n\n"
    f"  ── Final State ──────────\n"
    f"  SoH (true) : {results['soh_true_%'].iloc[-1]:.1f} %\n"
    f"  SoH (pred) : {results['soh_pred_%'].iloc[-1]:.1f} %\n\n"
    f"  ── Remaining Life ───────\n"
    f"  RUL (true) : {rul_days_true} days\n"
    f"  RUL (pred) : {rul_days_pred} days\n\n"
    f"  ── Anomalies ────────────\n"
    f"  Detected : {results['is_anomaly'].sum()} cycles\n"
    f"  (Z > {ANOMALY_Z}σ threshold)"
)
ax6.text(0.05, 0.95, kpi_text, transform=ax6.transAxes,
         fontsize=9, verticalalignment='top',
         fontfamily='monospace', color='#e0e0e0',
         bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='#4fc3f7', linewidth=1.5))

plt.suptitle('🔋 LSTM Battery Degradation — 2-Year Simulation Dashboard',
             fontsize=15, fontweight='bold', color='white', y=1.01)
plt.savefig(os.path.join(PLOT_DIR, '20_2yr_master_dashboard.png'),
            dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
plt.close()
print("  → Saved: outputs/plots/20_2yr_master_dashboard.png")


# ── PLOT B: Monthly Average Capacity (bar chart) ──
print("[PLOT] B — Monthly average capacity ...")
results['month'] = pd.to_datetime(results['date']).dt.to_period('M')
monthly = results.groupby('month')[['capacity_true', 'capacity_pred']].mean().reset_index()
monthly['month_str'] = monthly['month'].astype(str)

x = np.arange(len(monthly))
w = 0.38
fig, ax = plt.subplots(figsize=(18, 6))
ax.bar(x - w/2, monthly['capacity_true'], w, color='steelblue',  alpha=0.85, label='True Avg Capacity')
ax.bar(x + w/2, monthly['capacity_pred'], w, color='tomato',     alpha=0.85, label='Predicted Avg Capacity')
ax.axhline(EOL_CAP, color='red', linestyle='--', linewidth=1.5, label='EOL (1.6 Ahr)')
ax.set_xticks(x)
ax.set_xticklabels(monthly['month_str'], rotation=45, ha='right', fontsize=7)
ax.set_title('Monthly Average Capacity — True vs Predicted (2 Years)', fontsize=13, fontweight='bold')
ax.set_xlabel('Month'); ax.set_ylabel('Capacity (Ahr)')
ax.legend(); ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '21_monthly_avg_capacity.png'), dpi=150)
plt.close()
print("  → Saved: outputs/plots/21_monthly_avg_capacity.png")


# ── PLOT C: Rolling SoH with confidence band ──
print("[PLOT] C — Rolling SoH with confidence band ...")
window = 14
soh_rolling    = results['soh_pred_%'].rolling(window, center=True)
soh_roll_mean  = soh_rolling.mean()
soh_roll_std   = soh_rolling.std()

fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(dates_dt, results['soh_true_%'], color='royalblue', linewidth=1.5, label='True SoH', alpha=0.8)
ax.plot(dates_dt, soh_roll_mean,         color='tomato',    linewidth=2.0, label='Predicted SoH (14d roll)')
ax.fill_between(dates_dt,
    soh_roll_mean - 2 * soh_roll_std,
    soh_roll_mean + 2 * soh_roll_std,
    alpha=0.2, color='tomato', label='±2σ confidence')
ax.axhline(80, color='darkred', linestyle='--', linewidth=2, label='EOL (80%)')
ax.set_title('State of Health (%) — 2-Year Rolling Forecast with Confidence Band',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Date'); ax.set_ylabel('SoH (%)')
ax.set_ylim([60, 115])
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '22_soh_rolling_confidence.png'), dpi=150)
plt.close()
print("  → Saved: outputs/plots/22_soh_rolling_confidence.png")


# ── PLOT D: Input Features over 2 years (6-panel) ──
print("[PLOT] D — Input feature profiles over 2 years ...")
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
axes = axes.flatten()
feature_plot = [
    ('mean_voltage',        'Mean Voltage (V)',         'royalblue'),
    ('min_voltage',         'Min Voltage (V)',          'steelblue'),
    ('mean_current',        'Mean Current (A)',         'seagreen'),
    ('mean_temp',           'Mean Temperature (°C)',    'darkorange'),
    ('max_temp',            'Max Temperature (°C)',     'tomato'),
    ('discharge_duration',  'Discharge Duration (s)',   'purple'),
]
all_dates = pd.to_datetime(df_syn['date'])
for ax, (col, lbl, col_c) in zip(axes, feature_plot):
    ax.plot(all_dates, df_syn[col], color=col_c, linewidth=1.2, alpha=0.85)
    ax.plot(all_dates, df_syn[col].rolling(14, center=True).mean(),
            color='black', linewidth=2, label='14-day avg')
    ax.set_title(lbl, fontweight='bold')
    ax.set_ylabel(lbl.split('(')[0].strip())
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=7)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

plt.suptitle('LSTM Input Feature Profiles — 2-Year Synthetic Battery Aging',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '23_input_features_2yr.png'), dpi=150)
plt.close()
print("  → Saved: outputs/plots/23_input_features_2yr.png")


# ─────────────────────────────────────────────
# FINAL SUMMARY PRINT
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  FINAL 2-YEAR INFERENCE SUMMARY")
print("=" * 65)
print(f"\n  Model input  : (batch, {SEQ_LEN}, 7) — 7 features per cycle")
print(f"  Features     : {FEATURE_COLS}")
print(f"\n  Cycles total : {TOTAL_CYCLES} ({TOTAL_CYCLES/365:.1f} years)")
print(f"  Sequences    : {len(Xs)} (after seq_len={SEQ_LEN} warm-up)")
print(f"\n  MAE          : {mae:.4f} Ahr")
print(f"  RMSE         : {rmse:.4f} Ahr")
print(f"  MAPE         : {mape:.2f} %")
print(f"\n  Final SoH (true)    : {results['soh_true_%'].iloc[-1]:.1f} %")
print(f"  Final SoH (pred)    : {results['soh_pred_%'].iloc[-1]:.1f} %")
print(f"  RUL true            : {rul_days_true} cycles / days to EOL")
print(f"  RUL predicted       : {rul_days_pred} cycles / days to EOL")
print(f"  Anomalies detected  : {results['is_anomaly'].sum()} cycles")
print(f"\n  CSVs → outputs/data/")
print(f"  Plots (4 new) → outputs/plots/  [20–23]")
print("\n✓ 2-year simulation complete!")
