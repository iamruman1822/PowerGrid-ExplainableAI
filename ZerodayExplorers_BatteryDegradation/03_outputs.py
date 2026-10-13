"""
03_outputs.py
=============
NASA Battery Dataset — Full LSTM Output Generation
Loads trained model + predictions, generates ALL output plots and CSVs:
  1. Predicted vs Actual Capacity (test set)
  2. Per-battery Predicted vs Actual
  3. State of Health (SoH %) — actual + predicted
  4. Remaining Useful Life (RUL) — cycles to EOL
  5. Degradation Rate per cycle
  6. Multi-step Future Capacity Forecast
  7. Anomaly Detection (prediction error spikes)
  8. Error Distribution Histogram
  9. Error Metrics Table
  10. Comprehensive battery summary CSV
  11. Confusion-style capacity-band classification
"""

import os, sys, json, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(DATA_DIR, 'outputs')
PLOT_DIR   = os.path.join(OUTPUT_DIR, 'plots')
DATA_OUT   = os.path.join(OUTPUT_DIR, 'data')
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(DATA_OUT, exist_ok=True)

NOMINAL_CAP = 2.0      # Ahr
EOL_SOH     = 0.80     # End-of-Life = 80% SoH
EOL_CAP     = NOMINAL_CAP * EOL_SOH   # 1.6 Ahr
ANOMALY_Z   = 2.0      # Z-score threshold for anomaly

# ─────────────────────────────────────────────
# LOAD ARTEFACTS
# ─────────────────────────────────────────────
def require(path, desc):
    if not os.path.exists(path):
        sys.exit(f"ERROR: {desc} not found at {path}\nRun earlier scripts first.")
    return path

model_path = require(os.path.join(DATA_DIR, 'lstm_battery_model.keras'), 'LSTM model')
scaler_X   = pickle.load(open(require(os.path.join(DATA_OUT, 'scaler_X.pkl'), 'scaler_X'), 'rb'))
scaler_y   = pickle.load(open(require(os.path.join(DATA_OUT, 'scaler_y.pkl'), 'scaler_y'), 'rb'))
meta       = json.load(open(os.path.join(DATA_OUT, 'model_metadata.json')))
test_df    = pd.read_csv(require(os.path.join(DATA_OUT, 'test_predictions.csv'), 'test_predictions'))
all_df     = pd.read_csv(require(os.path.join(DATA_OUT, 'all_discharge_cycles.csv'), 'all_discharge'))

model = keras.models.load_model(model_path)

SEQ_LEN      = meta['seq_len']
FEATURE_COLS = meta['feature_cols']
print(f"Model loaded — seq_len={SEQ_LEN}, features={FEATURE_COLS}")
print(f"Test set: MAE={meta['test_mae']:.4f} Ahr, RMSE={meta['test_rmse']:.4f} Ahr, MAPE={meta['test_mape']:.2f}%")


# ─────────────────────────────────────────────
# HELPER: predict full capacity sequence per battery
# ─────────────────────────────────────────────
# Normalise cycle_num per battery
all_df['cycle_num_norm'] = all_df.groupby('battery')['cycle_num'].transform(
    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
)
all_df = all_df.dropna(subset=FEATURE_COLS + ['capacity']).copy()
all_df = all_df[all_df['capacity'] > 0].copy()

def predict_battery_sequence(battery_id):
    """Return DataFrame: cycle_num, true_cap, pred_cap for one battery."""
    sub = all_df[all_df['battery'] == battery_id].sort_values('cycle_num').reset_index(drop=True)
    if len(sub) <= SEQ_LEN:
        return pd.DataFrame()
    X_raw = sub[FEATURE_COLS].values
    y_raw = sub[['capacity']].values
    X_sc  = scaler_X.transform(X_raw)
    y_sc  = scaler_y.transform(y_raw)
    Xs, ys, cycle_nums = [], [], []
    for i in range(SEQ_LEN, len(X_sc)):
        Xs.append(X_sc[i - SEQ_LEN : i])
        ys.append(y_sc[i, 0])
        cycle_nums.append(sub['cycle_num'].iloc[i])
    Xs = np.array(Xs, dtype=np.float32)
    pred_sc = model.predict(Xs, verbose=0).flatten()
    pred    = scaler_y.inverse_transform(pred_sc.reshape(-1,1)).flatten()
    true    = scaler_y.inverse_transform(np.array(ys).reshape(-1,1)).flatten()
    return pd.DataFrame({'cycle_num': cycle_nums, 'true_cap': true, 'pred_cap': pred})


batteries = sorted(all_df['battery'].unique())
colors    = cm.tab20(np.linspace(0, 1, len(batteries)))

# Build full predictions for every battery
all_preds = {}
for bat in batteries:
    df_b = predict_battery_sequence(bat)
    if not df_b.empty:
        all_preds[bat] = df_b


# ─────────────────────────────────────────────
# PLOT 10: Predicted vs Actual — All Test Samples
# ─────────────────────────────────────────────
print("[PLOT] 10 — Predicted vs Actual (test scatter) ...")
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(test_df['true_cap'], test_df['pred_cap'],
           alpha=0.5, s=12, c='royalblue', label='Test predictions')
lims = [min(test_df['true_cap'].min(), test_df['pred_cap'].min()) - 0.05,
        max(test_df['true_cap'].max(), test_df['pred_cap'].max()) + 0.05]
ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect prediction')
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('Actual Capacity (Ahr)', fontsize=12)
ax.set_ylabel('Predicted Capacity (Ahr)', fontsize=12)
ax.set_title('Predicted vs Actual Capacity (Test Set)', fontsize=14, fontweight='bold')
ax.annotate(f"MAE={meta['test_mae']:.4f} Ahr\nRMSE={meta['test_rmse']:.4f} Ahr\nMAPE={meta['test_mape']:.2f}%",
            xy=(0.05, 0.92), xycoords='axes fraction',
            fontsize=11, bbox=dict(boxstyle='round', fc='lightyellow', ec='orange'))
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '10_pred_vs_actual_scatter.png'), dpi=150)
plt.close()
print("  → Saved: 10_pred_vs_actual_scatter.png")


# ─────────────────────────────────────────────
# PLOT 11: Per-Battery Predicted vs Actual (time series)
# ─────────────────────────────────────────────
print("[PLOT] 11 — Per-battery predicted vs actual capacity timeline ...")
n_bats = len(all_preds)
n_cols = 3
n_rows = (n_bats + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
axes = axes.flatten()

for idx, (bat, df_b) in enumerate(all_preds.items()):
    ax = axes[idx]
    ax.plot(df_b['cycle_num'], df_b['true_cap'], 'b-o', markersize=2.5, linewidth=1.5, label='Actual')
    ax.plot(df_b['cycle_num'], df_b['pred_cap'], 'r--^', markersize=2.5, linewidth=1.5, label='Predicted')
    ax.axhline(EOL_CAP, color='gray', linestyle=':', linewidth=1.2, label='EOL (80% SoH)')
    ax.set_title(f'Battery {bat}', fontweight='bold', fontsize=10)
    ax.set_xlabel('Cycle #'); ax.set_ylabel('Capacity (Ahr)')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

for j in range(idx + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Per-Battery: Actual vs Predicted Capacity', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '11_per_battery_pred_vs_actual.png'), dpi=150)
plt.close()
print("  → Saved: 11_per_battery_pred_vs_actual.png")


# ─────────────────────────────────────────────
# COMPUTE: SoH, RUL, Degradation Rate, Anomalies
# ─────────────────────────────────────────────
summary_rows = []

for bat, df_b in all_preds.items():
    if df_b.empty: continue

    # --- SoH ---
    cap_init       = df_b['true_cap'].iloc[0]
    df_b['soh_actual']    = (df_b['true_cap'] / cap_init) * 100
    df_b['soh_predicted'] = (df_b['pred_cap'] / cap_init) * 100

    # --- Degradation Rate (Ahr/cycle rolling) ---
    df_b = df_b.sort_values('cycle_num').reset_index(drop=True)
    df_b['deg_rate_actual']    = df_b['true_cap'].diff() / df_b['cycle_num'].diff()
    df_b['deg_rate_predicted'] = df_b['pred_cap'].diff() / df_b['cycle_num'].diff()
    # Smooth
    df_b['deg_rate_actual_smooth']    = df_b['deg_rate_actual'].rolling(5, center=True).mean()
    df_b['deg_rate_predicted_smooth'] = df_b['deg_rate_predicted'].rolling(5, center=True).mean()

    # --- Anomaly Detection on prediction error ---
    df_b['pred_error'] = df_b['pred_cap'] - df_b['true_cap']
    err_mean = df_b['pred_error'].mean()
    err_std  = df_b['pred_error'].std() + 1e-8
    df_b['error_z']   = (df_b['pred_error'] - err_mean) / err_std
    df_b['is_anomaly'] = df_b['error_z'].abs() > ANOMALY_Z

    # --- RUL (cycles until predicted cap < EOL_CAP) ---
    rul_pred, rul_actual = np.nan, np.nan
    below_pred = df_b[df_b['pred_cap'] < EOL_CAP]
    below_actual = df_b[df_b['true_cap'] < EOL_CAP]
    cur_cycle  = df_b['cycle_num'].max()

    if not below_pred.empty:
        eol_cycle_pred = below_pred['cycle_num'].min()
        rul_pred = max(0, eol_cycle_pred - cur_cycle)
    if not below_actual.empty:
        eol_cycle_actual = below_actual['cycle_num'].min()
        rul_actual = max(0, eol_cycle_actual - cur_cycle)

    # Avg degradation rate (last 10 cycles) for forecasting RUL if not yet at EOL
    if np.isnan(rul_pred) and len(df_b) >= 10:
        last10 = df_b.tail(10)
        rate = (last10['pred_cap'].iloc[-1] - last10['pred_cap'].iloc[0]) / (
            last10['cycle_num'].iloc[-1] - last10['cycle_num'].iloc[0] + 1e-8)
        cur_cap = df_b['pred_cap'].iloc[-1]
        if rate < 0:
            rul_pred = round((cur_cap - EOL_CAP) / abs(rate))

    soh_cur_actual = df_b['soh_actual'].iloc[-1]
    soh_cur_pred   = df_b['soh_predicted'].iloc[-1]

    summary_rows.append({
        'battery'              : bat,
        'total_cycles'         : int(df_b['cycle_num'].max()),
        'initial_capacity_Ahr' : round(cap_init, 4),
        'final_actual_cap_Ahr' : round(df_b['true_cap'].iloc[-1], 4),
        'final_pred_cap_Ahr'   : round(df_b['pred_cap'].iloc[-1], 4),
        'soh_actual_%'         : round(soh_cur_actual, 2),
        'soh_predicted_%'      : round(soh_cur_pred, 2),
        'rul_predicted_cycles' : int(rul_pred) if not np.isnan(rul_pred) else 'EOL reached',
        'rul_actual_cycles'    : int(rul_actual) if not np.isnan(rul_actual) else 'EOL reached',
        'avg_deg_rate_Ahr_cycle': round(float(df_b['deg_rate_actual'].dropna().mean()), 6),
        'anomaly_count'        : int(df_b['is_anomaly'].sum()),
        'mae_Ahr'              : round(float(np.mean(np.abs(df_b['true_cap'] - df_b['pred_cap']))), 5),
        'rmse_Ahr'             : round(float(np.sqrt(np.mean((df_b['true_cap'] - df_b['pred_cap'])**2))), 5),
    })

    # Attach back
    all_preds[bat] = df_b

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(DATA_OUT, 'battery_full_summary.csv'), index=False)
print("\nSummary table:")
print(summary_df.to_string(index=False))
print("\n[CSV] outputs/data/battery_full_summary.csv")


# ─────────────────────────────────────────────
# PLOT 12: SoH (%) — Actual + Predicted
# ─────────────────────────────────────────────
print("[PLOT] 12 — State of Health (SoH %) ...")
n_rows2 = (len(all_preds) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows2, n_cols, figsize=(18, n_rows2 * 4))
axes = axes.flatten()
for idx, (bat, df_b) in enumerate(all_preds.items()):
    ax = axes[idx]
    ax.plot(df_b['cycle_num'], df_b['soh_actual'],    'b-',  linewidth=2, label='Actual SoH')
    ax.plot(df_b['cycle_num'], df_b['soh_predicted'], 'r--', linewidth=2, label='Predicted SoH')
    ax.axhline(80, color='red', linestyle=':', linewidth=1.5, label='EOL=80%')
    ax.set_ylim([0, 115])
    ax.set_title(f'Battery {bat} — SoH', fontweight='bold', fontsize=10)
    ax.set_xlabel('Cycle #'); ax.set_ylabel('SoH (%)')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
for j in range(idx+1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle('State of Health (SoH %) — Actual vs Predicted', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '12_soh_per_battery.png'), dpi=150)
plt.close()
print("  → Saved: 12_soh_per_battery.png")


# ─────────────────────────────────────────────
# PLOT 13: Degradation Rate
# ─────────────────────────────────────────────
print("[PLOT] 13 — Degradation Rate per battery ...")
n_rows3 = (len(all_preds) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows3, n_cols, figsize=(18, n_rows3 * 4))
axes = axes.flatten()
for idx, (bat, df_b) in enumerate(all_preds.items()):
    ax = axes[idx]
    ax.plot(df_b['cycle_num'], df_b['deg_rate_actual_smooth'],    'b-',  linewidth=1.8, label='Actual (smooth)')
    ax.plot(df_b['cycle_num'], df_b['deg_rate_predicted_smooth'], 'r--', linewidth=1.8, label='Predicted (smooth)')
    ax.axhline(0, color='k', linewidth=0.7)
    ax.set_title(f'Battery {bat} — Degradation Rate', fontweight='bold', fontsize=10)
    ax.set_xlabel('Cycle #'); ax.set_ylabel('ΔAhr/Cycle')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
for j in range(idx+1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle('Degradation Rate (ΔCapacity/ΔCycle) — Rolling Mean', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '13_degradation_rate.png'), dpi=150)
plt.close()
print("  → Saved: 13_degradation_rate.png")


# ─────────────────────────────────────────────
# PLOT 14: Anomaly Detection
# ─────────────────────────────────────────────
print("[PLOT] 14 — Anomaly Detection ...")
n_rows4 = (len(all_preds) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows4, n_cols, figsize=(18, n_rows4 * 4))
axes = axes.flatten()
for idx, (bat, df_b) in enumerate(all_preds.items()):
    ax = axes[idx]
    normal  = df_b[~df_b['is_anomaly']]
    anomaly = df_b[ df_b['is_anomaly']]
    ax.plot(normal['cycle_num'],  normal['pred_error'],  'b.',  markersize=4, label='Normal')
    ax.scatter(anomaly['cycle_num'], anomaly['pred_error'], c='red', s=40, zorder=5, label=f'Anomaly ({len(anomaly)})')
    ax.axhline(0, color='k', linewidth=0.8)
    ax.axhline( ANOMALY_Z * df_b['pred_error'].std(), color='orange', linestyle='--', linewidth=1, label=f'±{ANOMALY_Z}σ')
    ax.axhline(-ANOMALY_Z * df_b['pred_error'].std(), color='orange', linestyle='--', linewidth=1)
    ax.set_title(f'Battery {bat} — Prediction Anomalies', fontweight='bold', fontsize=9)
    ax.set_xlabel('Cycle #'); ax.set_ylabel('Prediction Error (Ahr)')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
for j in range(idx+1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle(f'Anomaly Detection (Z-score > {ANOMALY_Z}) in Prediction Errors', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '14_anomaly_detection.png'), dpi=150)
plt.close()
print("  → Saved: 14_anomaly_detection.png")


# ─────────────────────────────────────────────
# PLOT 15: Multi-Step Future Forecast (20 cycles ahead)
# ─────────────────────────────────────────────
print("[PLOT] 15 — Multi-step future capacity forecast ...")
FORECAST_STEPS = 20

def forecast_future(battery_id, steps=20):
    """Auto-regressive multi-step forecast from last known sequence."""
    sub = all_df[all_df['battery'] == battery_id].sort_values('cycle_num').reset_index(drop=True)
    if len(sub) <= SEQ_LEN: return [], []
    X_raw = sub[FEATURE_COLS].values
    X_sc  = scaler_X.transform(X_raw)
    # Seed = last SEQ_LEN steps
    seed  = X_sc[-SEQ_LEN:].copy()   # (SEQ_LEN, n_feat)
    last_cycle = sub['cycle_num'].max()
    preds, cycle_nums = [], []
    seq = seed.copy()
    for s in range(steps):
        inp  = seq.reshape(1, SEQ_LEN, -1)
        pred_sc = model.predict(inp, verbose=0)[0, 0]
        pred_cap = float(scaler_y.inverse_transform([[pred_sc]])[0, 0])
        preds.append(pred_cap)
        cycle_nums.append(last_cycle + s + 1)
        # Shift window: update next row, keeping feature values same as last row
        new_row = seq[-1].copy()
        # Update capacity feature indirectly via cycle_num_norm (feature index 6)
        # Increment cycle_num_norm slightly
        new_row[FEATURE_COLS.index('cycle_num_norm')] = min(1.0,
            seq[-1, FEATURE_COLS.index('cycle_num_norm')] + 1 / (len(sub) + steps))
        seq = np.vstack([seq[1:], new_row])
    return cycle_nums, preds

fig, ax = plt.subplots(figsize=(14, 7))
for bat, col in zip(batteries, colors):
    if bat not in all_preds: continue
    df_b = all_preds[bat]
    ax.plot(df_b['cycle_num'], df_b['true_cap'], color=col, linewidth=1.5, alpha=0.7)
    fut_cyc, fut_cap = forecast_future(bat, FORECAST_STEPS)
    if fut_cyc:
        ax.plot(fut_cyc, fut_cap, color=col, linewidth=2.0, linestyle='--',
                marker='D', markersize=3, label=bat)

ax.axhline(EOL_CAP, color='red', linestyle=':', linewidth=2, label='EOL threshold (1.6 Ahr)')
ax.set_title(f'Multi-Step Future Forecast (+{FORECAST_STEPS} cycles) — All Batteries', fontsize=14, fontweight='bold')
ax.set_xlabel('Cycle Number', fontsize=12)
ax.set_ylabel('Capacity (Ahr)', fontsize=12)
ax.legend(fontsize=7, ncol=4); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '15_future_forecast.png'), dpi=150)
plt.close()
print("  → Saved: 15_future_forecast.png")


# ─────────────────────────────────────────────
# PLOT 16: RUL Comparison (predicted vs actual)
# ─────────────────────────────────────────────
print("[PLOT] 16 — RUL comparison ...")
rul_data = summary_df[['battery','rul_predicted_cycles','rul_actual_cycles']].copy()
rul_data['rul_predicted_cycles'] = pd.to_numeric(rul_data['rul_predicted_cycles'], errors='coerce').fillna(0)
rul_data['rul_actual_cycles']    = pd.to_numeric(rul_data['rul_actual_cycles'], errors='coerce').fillna(0)

x = np.arange(len(rul_data))
w = 0.35
fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - w/2, rul_data['rul_actual_cycles'],   w, label='RUL Actual',    color='steelblue')
bars2 = ax.bar(x + w/2, rul_data['rul_predicted_cycles'], w, label='RUL Predicted', color='tomato', alpha=0.85)
ax.bar_label(bars1, padding=2, fontsize=8)
ax.bar_label(bars2, padding=2, fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(rul_data['battery'], rotation=30, ha='right')
ax.set_title('Remaining Useful Life (RUL) — Predicted vs Actual', fontsize=14, fontweight='bold')
ax.set_xlabel('Battery'); ax.set_ylabel('Remaining Cycles to EOL')
ax.legend(); ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '16_rul_comparison.png'), dpi=150)
plt.close()
print("  → Saved: 16_rul_comparison.png")


# ─────────────────────────────────────────────
# PLOT 17: Error Distribution Histogram
# ─────────────────────────────────────────────
print("[PLOT] 17 — Prediction error distribution ...")
all_errors = []
for bat, df_b in all_preds.items():
    all_errors.extend((df_b['pred_cap'] - df_b['true_cap']).tolist())
all_errors = np.array(all_errors)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(all_errors, bins=50, color='royalblue', edgecolor='navy', alpha=0.8)
axes[0].axvline(0, color='red', linewidth=2, linestyle='--', label='Zero error')
axes[0].axvline(all_errors.mean(), color='green', linewidth=2, label=f'Mean={all_errors.mean():.4f}')
axes[0].set_title('Prediction Error Distribution', fontweight='bold')
axes[0].set_xlabel('Error (Ahr)'); axes[0].set_ylabel('Frequency')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].hist(np.abs(all_errors), bins=50, color='darkorange', edgecolor='saddlebrown', alpha=0.8)
axes[1].axvline(np.abs(all_errors).mean(), color='red', linewidth=2, label=f'Mean AE={np.abs(all_errors).mean():.4f}')
axes[1].set_title('Absolute Error Distribution', fontweight='bold')
axes[1].set_xlabel('|Error| (Ahr)'); axes[1].set_ylabel('Frequency')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.suptitle('LSTM Prediction Error Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '17_error_distribution.png'), dpi=150)
plt.close()
print("  → Saved: 17_error_distribution.png")


# ─────────────────────────────────────────────
# PLOT 18: Global Capacity — All Batteries (actual + predicted overlay)
# ─────────────────────────────────────────────
print("[PLOT] 18 — Global capacity: actual (solid) + predicted (dashed) overlay ...")
fig, ax = plt.subplots(figsize=(14, 7))
for (bat, df_b), col in zip(all_preds.items(), colors):
    ax.plot(df_b['cycle_num'], df_b['true_cap'], '-',  linewidth=1.5, color=col, alpha=0.8)
    ax.plot(df_b['cycle_num'], df_b['pred_cap'], '--', linewidth=1.2, color=col, alpha=0.6)

ax.axhline(EOL_CAP, color='red', linestyle=':', linewidth=2, label='EOL (80% SoH = 1.6 Ahr)')
from matplotlib.lines import Line2D
legend_elem = [Line2D([0],[0], color='gray',   linewidth=2, linestyle='-',  label='Actual capacity'),
               Line2D([0],[0], color='gray',   linewidth=2, linestyle='--', label='Predicted capacity'),
               Line2D([0],[0], color='red',    linewidth=2, linestyle=':',  label='EOL threshold')]
ax.legend(handles=legend_elem, loc='upper right', fontsize=10)
ax.set_title('All Batteries: Actual (solid) vs LSTM Predicted (dashed) Capacity', fontsize=14, fontweight='bold')
ax.set_xlabel('Discharge Cycle Number', fontsize=12)
ax.set_ylabel('Capacity (Ahr)', fontsize=12)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '18_global_actual_vs_predicted.png'), dpi=150)
plt.close()
print("  → Saved: 18_global_actual_vs_predicted.png")


# ─────────────────────────────────────────────
# PLOT 19: Error Metrics Table (per battery)
# ─────────────────────────────────────────────
print("[PLOT] 19 — Error metrics heatmap ...")
metrics_df = summary_df[['battery','mae_Ahr','rmse_Ahr','soh_actual_%','soh_predicted_%']].set_index('battery')
fig, ax = plt.subplots(figsize=(10, max(4, len(metrics_df) * 0.5 + 2)))
sns.heatmap(metrics_df[['mae_Ahr','rmse_Ahr']].astype(float),
            annot=True, fmt='.4f', cmap='YlOrRd', ax=ax,
            linewidths=0.5, cbar_kws={'label': 'Error (Ahr)'})
ax.set_title('Per-Battery LSTM Error Metrics (MAE & RMSE)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '19_error_metrics_heatmap.png'), dpi=150)
plt.close()
print("  → Saved: 19_error_metrics_heatmap.png")


# ─────────────────────────────────────────────
# SAVE: Full per-cycle predictions CSV
# ─────────────────────────────────────────────
all_preds_list = []
for bat, df_b in all_preds.items():
    df_b['battery'] = bat
    all_preds_list.append(df_b)
full_preds_df = pd.concat(all_preds_list, ignore_index=True)
full_preds_df.to_csv(os.path.join(DATA_OUT, 'full_cycle_predictions.csv'), index=False)
print("[CSV] outputs/data/full_cycle_predictions.csv")

# ─────────────────────────────────────────────
# PRINT: Final Summary
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("  LSTM BATTERY DEGRADATION MODEL — FINAL RESULTS SUMMARY")
print("=" * 70)
print(f"\n  Model: 3-Layer Stacked LSTM (128 → 64 → 32 units)")
print(f"  Sequence Length: {SEQ_LEN} cycles")
print(f"  Features: {FEATURE_COLS}")
print(f"\n  OVERALL TEST SET METRICS:")
print(f"    MAE  = {meta['test_mae']:.4f} Ahr")
print(f"    RMSE = {meta['test_rmse']:.4f} Ahr")
print(f"    MAPE = {meta['test_mape']:.2f} %")
print(f"\n  EPOCHS TRAINED: {meta['epochs_run']}")
print(f"  BEST VAL LOSS : {meta['best_val_loss']}")
print(f"\n  PER-BATTERY SUMMARY:")
print(summary_df[['battery','total_cycles','soh_actual_%','soh_predicted_%',
                  'rul_predicted_cycles','anomaly_count','mae_Ahr']].to_string(index=False))
print("\n" + "=" * 70)
print("\n✓ All outputs generated!")
print(f"  Plots  ({len(os.listdir(PLOT_DIR))}) → outputs/plots/")
print(f"  CSVs   → outputs/data/")
