"""
05_nasa_2yr_inference.py
========================
Uses the REAL NASA battery discharge data as input to the trained LSTM.
For each battery:
  1. Runs LSTM inference on ALL real recorded cycles
  2. If total cycles < 730, auto-regressively forecasts forward
     using the real tail as seed until 730 cycles (2 years) are reached
  3. Maps cycle numbers to calendar dates (1 cycle = 1 day, start Jan 1 2024)
  4. Produces full 2-year outputs: capacity, SoH, RUL, degradation rate,
     anomaly flags, cumulative fade, 30-day further forecast
  5. Saves per-battery CSVs and a combined master CSV
  6. Generates a comprehensive set of output plots

All outputs → outputs/plots/ (plots 24+) and outputs/data/nasa_2yr_*.csv
"""

import os, sys, pickle, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.cm as cm
from datetime import date, timedelta
import tensorflow as tf
from tensorflow import keras
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(DATA_DIR, 'outputs')
PLOT_DIR   = os.path.join(OUTPUT_DIR, 'plots')
DATA_OUT   = os.path.join(OUTPUT_DIR, 'data')
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(DATA_OUT, exist_ok=True)

TARGET_CYCLES  = 730          # 2 years @ 1 cycle/day
FORECAST_EXTRA = 30           # extra days beyond 2 years
START_DATE     = date(2024, 1, 1)
EOL_CAP        = 1.6          # Ahr (80% of 2.0 Ahr nominal)
NOMINAL_CAP    = 2.0
ANOMALY_Z      = 2.0
SEQ_LEN        = 10

# ─────────────────────────────────────────────
# LOAD MODEL + SCALERS + METADATA
# ─────────────────────────────────────────────
def require(path, name):
    if not os.path.exists(path):
        sys.exit(f"ERROR: {name} not found — run earlier scripts first.")
    return path

model    = keras.models.load_model(require(os.path.join(DATA_DIR, 'lstm_battery_model.keras'), 'LSTM model'))
scaler_X = pickle.load(open(require(os.path.join(DATA_OUT, 'scaler_X.pkl'), 'scaler_X'), 'rb'))
scaler_y = pickle.load(open(require(os.path.join(DATA_OUT, 'scaler_y.pkl'), 'scaler_y'), 'rb'))
meta     = json.load(open(os.path.join(DATA_OUT, 'model_metadata.json')))
FEATURE_COLS = meta['feature_cols']

print("=" * 68)
print("  LSTM INFERENCE — REAL NASA DATA → 2-YEAR OUTPUT PER BATTERY")
print("=" * 68)
print(f"\nModel input shape : (batch, {SEQ_LEN}, {len(FEATURE_COLS)})")
print(f"Target            : 2 years = {TARGET_CYCLES} cycles per battery")
print(f"Features          : {FEATURE_COLS}\n")

# ─────────────────────────────────────────────
# LOAD REAL NASA DISCHARGE DATA
# ─────────────────────────────────────────────
csv_path = os.path.join(DATA_OUT, 'all_discharge_cycles.csv')
if not os.path.exists(csv_path):
    sys.exit("ERROR: Run 01_data_analysis.py first to build all_discharge_cycles.csv")

df_all = pd.read_csv(csv_path)

# Add cycle_num_norm per battery (matches training)
df_all['cycle_num_norm'] = df_all.groupby('battery')['cycle_num'].transform(
    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
)
df_all = df_all.dropna(subset=FEATURE_COLS + ['capacity']).copy()
df_all = df_all[df_all['capacity'] > 0].copy()

batteries = sorted(df_all['battery'].unique())
print(f"Batteries found : {batteries}\n")

# ─────────────────────────────────────────────
# CORE FUNCTION: predict + extend to 2 years
# ─────────────────────────────────────────────
def process_battery(battery_id):
    sub = df_all[df_all['battery'] == battery_id].sort_values('cycle_num').reset_index(drop=True)
    n_real = len(sub)
    print(f"  {battery_id}: {n_real} real cycles", end='')

    X_raw = sub[FEATURE_COLS].values
    y_raw = sub[['capacity']].values
    X_sc  = scaler_X.transform(X_raw)
    y_sc  = scaler_y.transform(y_raw)

    # ── PART 1: inference on real recorded cycles ──
    real_pred_caps, real_true_caps, real_cycles = [], [], []
    for i in range(SEQ_LEN, n_real):
        seq     = X_sc[i - SEQ_LEN : i].reshape(1, SEQ_LEN, -1)
        pred_sc = model.predict(seq, verbose=0)[0, 0]
        pred_cap = float(scaler_y.inverse_transform([[pred_sc]])[0, 0])
        true_cap = float(y_raw[i, 0])
        real_pred_caps.append(pred_cap)
        real_true_caps.append(true_cap)
        real_cycles.append(int(sub['cycle_num'].iloc[i]))

    # ── PART 2: BATCHED forecast to fill to TARGET_CYCLES ──
    n_to_forecast = max(0, TARGET_CYCLES - n_real)
    print(f" → {n_to_forecast} cycles forecasted to reach 2 years")

    forecast_caps, forecast_cycles = [], []
    if n_to_forecast > 0:
        seed_seq  = X_sc[-SEQ_LEN:].copy()   # (SEQ_LEN, n_feat)
        last_norm = float(sub['cycle_num_norm'].iloc[-1])
        if len(sub) > 10:
            norm_inc = (sub['cycle_num_norm'].iloc[-1] - sub['cycle_num_norm'].iloc[-10]) / 10
        else:
            norm_inc = 1.0 / max(TARGET_CYCLES, 1)
        norm_inc = abs(norm_inc) if norm_inc != 0 else 1 / TARGET_CYCLES
        last_cycle = int(sub['cycle_num'].max())

        # ── BUILD ALL ROLLING SEQUENCES IN NUMPY (no per-step predict) ──
        # Extend the feature array by n_to_forecast rows first,
        # each new row = previous last row with updated cycle_num_norm
        extended = seed_seq.copy()   # starts as (SEQ_LEN, n_feat)
        norm_idx = FEATURE_COLS.index('cycle_num_norm')
        extra_rows = []
        for step in range(n_to_forecast):
            new_row = extended[-1].copy()
            new_row[norm_idx] = min(1.0, last_norm + norm_inc * (step + 1))
            extra_rows.append(new_row)
        extra_arr = np.array(extra_rows, dtype=np.float32)   # (n_to_forecast, n_feat)

        # Full padded array: seed window + all future rows
        full_arr = np.vstack([seed_seq, extra_arr])  # (SEQ_LEN + n_to_forecast, n_feat)

        # Build all sequences as a batch: shape (n_to_forecast, SEQ_LEN, n_feat)
        all_seqs = np.stack(
            [full_arr[i : i + SEQ_LEN] for i in range(n_to_forecast)],
            axis=0
        ).astype(np.float32)

        # SINGLE batched predict call
        preds_sc  = model.predict(all_seqs, verbose=0, batch_size=256).flatten()
        preds_cap = scaler_y.inverse_transform(preds_sc.reshape(-1, 1)).flatten()
        preds_cap = np.clip(preds_cap, 0.3, NOMINAL_CAP)

        forecast_caps   = preds_cap.tolist()
        forecast_cycles = [last_cycle + s + 1 for s in range(n_to_forecast)]

    # ── ASSEMBLE FULL 2-YEAR RESULT ──
    all_cycles = real_cycles + forecast_cycles
    all_pred   = real_pred_caps + forecast_caps
    all_true   = real_true_caps + [np.nan] * len(forecast_cycles)
    is_forecast= [False] * len(real_cycles) + [True] * len(forecast_cycles)

    # Calendar dates: cycle 1 of each battery = START_DATE
    first_cycle = sub['cycle_num'].min()
    all_dates   = [START_DATE + timedelta(days=int(c - first_cycle)) for c in all_cycles]

    df_out = pd.DataFrame({
        'battery'      : battery_id,
        'date'         : all_dates,
        'cycle_num'    : all_cycles,
        'capacity_true': all_true,
        'capacity_pred': all_pred,
        'is_forecast'  : is_forecast,
    })

    # SoH
    df_out['soh_true_%'] = (df_out['capacity_true'] / NOMINAL_CAP) * 100
    df_out['soh_pred_%'] = (df_out['capacity_pred'] / NOMINAL_CAP) * 100

    # Degradation rate
    df_out['deg_rate_pred'] = df_out['capacity_pred'].diff() / df_out['cycle_num'].diff()
    df_out['deg_rate_true'] = df_out['capacity_true'].diff() / df_out['cycle_num'].diff()
    df_out['deg_rate_pred_smooth'] = df_out['deg_rate_pred'].rolling(7, center=True, min_periods=3).mean()
    df_out['deg_rate_true_smooth'] = df_out['deg_rate_true'].rolling(7, center=True, min_periods=3).mean()

    # Prediction error & anomalies (only on real cycles)
    real_mask = ~df_out['is_forecast']
    df_out['pred_error'] = np.nan
    df_out.loc[real_mask, 'pred_error'] = (
        df_out.loc[real_mask, 'capacity_pred'] - df_out.loc[real_mask, 'capacity_true']
    )
    err_vals = df_out['pred_error'].dropna()
    if len(err_vals) > 1:
        emean, estd = err_vals.mean(), err_vals.std() + 1e-8
        df_out['error_z']    = (df_out['pred_error'] - emean) / estd
        df_out['is_anomaly'] = df_out['error_z'].abs() > ANOMALY_Z
    else:
        df_out['error_z'] = np.nan
        df_out['is_anomaly'] = False

    # Cumulative fade
    first_pred = df_out['capacity_pred'].iloc[0]
    df_out['cumulative_fade_Ahr'] = first_pred - df_out['capacity_pred']

    return df_out


# ─────────────────────────────────────────────
# RUN FOR ALL BATTERIES
# ─────────────────────────────────────────────
all_results = {}
for bat in batteries:
    try:
        df_bat = process_battery(bat)
        all_results[bat] = df_bat
    except Exception as e:
        print(f"  [ERROR] {bat}: {e}")

# Combine
master_df = pd.concat(all_results.values(), ignore_index=True)
master_df.to_csv(os.path.join(DATA_OUT, 'nasa_2yr_all_batteries.csv'), index=False)
print(f"\n[CSV] nasa_2yr_all_batteries.csv  ({len(master_df)} total rows)")

# ─────────────────────────────────────────────
# COMPUTE PER-BATTERY SUMMARY
# ─────────────────────────────────────────────
def compute_rul(cap_series, cycle_series, eol=EOL_CAP):
    below = cap_series[cap_series < eol]
    if not below.empty:
        eol_c = cycle_series.iloc[below.index[0]]
        return max(0, int(eol_c - cycle_series.iloc[-1]))
    last_n = min(30, len(cap_series))
    x, y = cycle_series.tail(last_n).values, cap_series.tail(last_n).values
    if len(x) < 2: return 9999
    slope = np.polyfit(x, y, 1)[0]
    if slope >= 0: return 9999
    return max(0, int((cap_series.iloc[-1] - eol) / abs(slope)))

summary_rows = []
for bat, df_b in all_results.items():
    real = df_b[~df_b['is_forecast']]
    mae  = float(real['pred_error'].abs().mean()) if not real.empty else np.nan
    rmse = float(np.sqrt((real['pred_error']**2).mean())) if not real.empty else np.nan
    rul  = compute_rul(df_b['capacity_pred'], df_b['cycle_num'])
    summary_rows.append({
        'battery'              : bat,
        'real_cycles'          : int((~df_b['is_forecast']).sum()),
        'forecast_cycles'      : int(df_b['is_forecast'].sum()),
        'total_2yr_cycles'     : len(df_b),
        'initial_cap_Ahr'      : round(df_b['capacity_pred'].iloc[0], 4),
        'final_cap_pred_Ahr'   : round(df_b['capacity_pred'].iloc[-1], 4),
        'final_soh_pred_%'     : round(df_b['soh_pred_%'].iloc[-1], 2),
        'rul_pred_cycles_days' : rul,
        'eol_reached'          : 'YES' if rul == 0 else 'NO',
        'anomaly_count'        : int(df_b['is_anomaly'].sum()),
        'avg_deg_rate_mAhr'    : round(float(df_b['deg_rate_pred'].dropna().mean() * 1000), 4),
        'mae_Ahr'              : round(mae, 5) if not np.isnan(mae) else 'N/A',
        'rmse_Ahr'             : round(rmse, 5) if not np.isnan(rmse) else 'N/A',
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(DATA_OUT, 'nasa_2yr_summary.csv'), index=False)
print("[CSV] nasa_2yr_summary.csv")
print("\n── Per-Battery 2-Year Summary ──────────────────────────────────────")
print(summary_df[['battery','real_cycles','forecast_cycles','final_soh_pred_%',
                  'rul_pred_cycles_days','eol_reached','anomaly_count','mae_Ahr']].to_string(index=False))


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────
colors_map = {bat: col for bat, col in
              zip(batteries, cm.tab20(np.linspace(0, 1, len(batteries))))}

# ── PLOT 24: Global 2-Year Capacity — All Batteries Overlay ──
print("\n[PLOT] 24 — Global 2-year capacity (real + forecast per battery) ...")
fig, ax = plt.subplots(figsize=(16, 8))

for bat, df_b in all_results.items():
    col = colors_map[bat]
    real_part = df_b[~df_b['is_forecast']]
    fore_part = df_b[ df_b['is_forecast']]

    dates_r = pd.to_datetime(real_part['date'])
    ax.plot(dates_r, real_part['capacity_pred'], color=col, linewidth=2.0,
            label=bat, alpha=0.9)

    if not fore_part.empty:
        dates_f = pd.to_datetime(fore_part['date'])
        ax.plot(dates_f, fore_part['capacity_pred'], color=col,
                linewidth=1.5, linestyle='--', alpha=0.6)
        # Mark transition point
        if not real_part.empty:
            ax.axvline(pd.to_datetime(real_part['date'].iloc[-1]),
                       color=col, linewidth=0.5, linestyle=':', alpha=0.4)

ax.axhline(EOL_CAP, color='red', linestyle='-.', linewidth=2,
           label='EOL threshold (1.6 Ahr)')
ax.set_title('LSTM Predicted Capacity — All NASA Batteries (2 Years)\n'
             'Solid = real data inference  |  Dashed = auto-regressive forecast',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Predicted Capacity (Ahr)', fontsize=12)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
ax.legend(fontsize=7, ncol=4, loc='upper right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '24_nasa_2yr_global_capacity.png'), dpi=150)
plt.close()
print("  → Saved: 24_nasa_2yr_global_capacity.png")


# ── PLOT 25: Per-Battery Predicted vs Actual + Forecast ──
print("[PLOT] 25 — Per-battery capacity: actual vs predicted vs forecast ...")
n_bats = len(all_results)
n_cols = 3
n_rows = (n_bats + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4.5))
axes = axes.flatten()

for idx, (bat, df_b) in enumerate(all_results.items()):
    ax = axes[idx]
    col = colors_map[bat]
    real = df_b[~df_b['is_forecast']]
    fore = df_b[ df_b['is_forecast']]
    dates_all  = pd.to_datetime(df_b['date'])
    dates_real = pd.to_datetime(real['date'])
    dates_fore = pd.to_datetime(fore['date'])

    # True capacity (real cycles only)
    if not real.empty:
        ax.plot(dates_real, real['capacity_true'], 'b-o', markersize=2,
                linewidth=1.5, label='NASA True', alpha=0.8)
    # Predicted on real
    if not real.empty:
        ax.plot(dates_real, real['capacity_pred'], color='tomato',
                linewidth=2, linestyle='--', label='LSTM Predicted')
    # Forecast extension
    if not fore.empty:
        ax.plot(dates_fore, fore['capacity_pred'], color='seagreen',
                linewidth=2, linestyle=':', marker='D', markersize=2,
                label=f'Forecast (+{len(fore)}d)')
        if not real.empty:
            ax.axvline(pd.to_datetime(real['date'].iloc[-1]), color='gray',
                       linewidth=1, linestyle='--', alpha=0.6, label='End of real data')

    ax.axhline(EOL_CAP, color='red', linestyle='-.', linewidth=1.2, alpha=0.7)
    ax.set_title(f'{bat}  (real:{len(real)} | fcast:{len(fore)})',
                 fontweight='bold', fontsize=9)
    ax.set_xlabel('Date', fontsize=8); ax.set_ylabel('Cap (Ahr)', fontsize=8)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=7)
    ax.legend(fontsize=6); ax.grid(alpha=0.3)

for j in range(idx + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Per-Battery: NASA True | LSTM Predicted | Forecast Extension (2 Years)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '25_nasa_2yr_per_battery.png'), dpi=150)
plt.close()
print("  → Saved: 25_nasa_2yr_per_battery.png")


# ── PLOT 26: SoH (%) — All Batteries 2 Years ──
print("[PLOT] 26 — SoH (%) all batteries 2 years ...")
fig, ax = plt.subplots(figsize=(16, 8))
for bat, df_b in all_results.items():
    col = colors_map[bat]
    real = df_b[~df_b['is_forecast']]
    fore = df_b[ df_b['is_forecast']]
    ax.plot(pd.to_datetime(real['date']), real['soh_pred_%'], color=col,
            linewidth=2, label=bat)
    if not fore.empty:
        ax.plot(pd.to_datetime(fore['date']), fore['soh_pred_%'], color=col,
                linewidth=1.5, linestyle='--', alpha=0.7)

ax.axhline(80, color='red', linestyle='-.', linewidth=2, label='EOL = 80% SoH')
ax.set_title('State of Health (%) — All NASA Batteries (2-Year LSTM Output)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Date'); ax.set_ylabel('SoH (%)')
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
ax.legend(ncol=4, fontsize=7); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '26_nasa_2yr_soh_all.png'), dpi=150)
plt.close()
print("  → Saved: 26_nasa_2yr_soh_all.png")


# ── PLOT 27: Degradation Rate — All Batteries ──
print("[PLOT] 27 — Degradation rate (2 years) ...")
fig, ax = plt.subplots(figsize=(16, 7))
for bat, df_b in all_results.items():
    col = colors_map[bat]
    ax.plot(pd.to_datetime(df_b['date']), df_b['deg_rate_pred_smooth'] * 1000,
            color=col, linewidth=1.5, alpha=0.85, label=bat)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_title('Predicted Degradation Rate (ΔCap/ΔCycle) — 7-Cycle Smoothed',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Date'); ax.set_ylabel('mAhr / cycle')
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
ax.legend(ncol=4, fontsize=7); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '27_nasa_2yr_degradation_rate.png'), dpi=150)
plt.close()
print("  → Saved: 27_nasa_2yr_degradation_rate.png")


# ── PLOT 28: RUL Bar Chart ──
print("[PLOT] 28 — RUL bar chart (2-year endpoint) ...")
fig, ax = plt.subplots(figsize=(12, 6))
rul_vals  = [r['rul_pred_cycles_days'] for r in summary_rows]
bat_names = [r['battery'] for r in summary_rows]
colors_bar = ['tomato' if r['eol_reached'] == 'YES' else 'steelblue' for r in summary_rows]
bars = ax.bar(bat_names, rul_vals, color=colors_bar, edgecolor='navy', alpha=0.85)
ax.bar_label(bars, padding=3, fontsize=9, fmt='%d')
ax.set_title('Remaining Useful Life (days/cycles) at End of 2-Year Forecast',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Battery'); ax.set_ylabel('RUL (cycles = days)')
ax.axhline(0, color='red', linewidth=1)
plt.xticks(rotation=30, ha='right')
ax.grid(alpha=0.3, axis='y')
from matplotlib.patches import Patch
legend_elem = [Patch(facecolor='steelblue', label='EOL not yet reached'),
               Patch(facecolor='tomato',    label='EOL reached within 2 years')]
ax.legend(handles=legend_elem)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '28_nasa_2yr_rul.png'), dpi=150)
plt.close()
print("  → Saved: 28_nasa_2yr_rul.png")


# ── PLOT 29: Anomaly Heatmap (battery × month) ──
print("[PLOT] 29 — Anomaly heatmap (battery × month) ...")
master_df['date_dt'] = pd.to_datetime(master_df['date'])
master_df['month'] = master_df['date_dt'].dt.to_period('M')
anom_pivot = master_df[master_df['is_anomaly']].groupby(
    ['battery','month']).size().unstack(fill_value=0)
# Align to all months
all_months = pd.period_range(START_DATE.strftime('%Y-%m'), periods=24, freq='M')
anom_pivot = anom_pivot.reindex(columns=all_months, fill_value=0)

fig, ax = plt.subplots(figsize=(20, max(5, len(all_results) * 0.7 + 2)))
import seaborn as sns
sns.heatmap(anom_pivot.astype(float), ax=ax, cmap='YlOrRd',
            linewidths=0.3, annot=True, fmt='.0f', cbar_kws={'label':'Anomaly count'})
ax.set_title('Anomaly Count per Battery per Month (Z-score > 2σ)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Month'); ax.set_ylabel('Battery')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '29_nasa_2yr_anomaly_heatmap.png'), dpi=150)
plt.close()
print("  → Saved: 29_nasa_2yr_anomaly_heatmap.png")


# ── PLOT 30: Cumulative Capacity Fade ──
print("[PLOT] 30 — Cumulative capacity fade ...")
fig, ax = plt.subplots(figsize=(16, 7))
for bat, df_b in all_results.items():
    col = colors_map[bat]
    ax.plot(pd.to_datetime(df_b['date']), df_b['cumulative_fade_Ahr'],
            color=col, linewidth=2, label=bat)
    if df_b['is_forecast'].any():
        fore_start = pd.to_datetime(df_b[df_b['is_forecast']]['date'].iloc[0])
        ax.axvline(fore_start, color=col, linewidth=0.5, linestyle=':', alpha=0.4)

ax.set_title('Cumulative Capacity Fade from Initial — All NASA Batteries (2 Years)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Date'); ax.set_ylabel('Capacity Lost (Ahr)')
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
ax.legend(ncol=4, fontsize=7); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '30_nasa_2yr_cumulative_fade.png'), dpi=150)
plt.close()
print("  → Saved: 30_nasa_2yr_cumulative_fade.png")


# ── PLOT 31: Dark-mode Master Dashboard ──
print("[PLOT] 31 — Dark-mode 2-year NASA inference dashboard ...")
fig = plt.figure(figsize=(22, 14))
fig.patch.set_facecolor('#0d1117')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

def dark_ax(ax):
    ax.set_facecolor('#161b22')
    for s in ax.spines.values(): s.set_color('#30363d')
    ax.tick_params(colors='#c9d1d9', labelsize=8)
    ax.xaxis.label.set_color('#c9d1d9'); ax.yaxis.label.set_color('#c9d1d9')
    ax.title.set_color('#e6edf3'); ax.grid(alpha=0.15, color='#c9d1d9')
    return ax

# Panel 1: Global Capacity
ax1 = dark_ax(fig.add_subplot(gs[0, :2]))
for bat, df_b in all_results.items():
    col = colors_map[bat]
    real = df_b[~df_b['is_forecast']]
    fore = df_b[ df_b['is_forecast']]
    ax1.plot(pd.to_datetime(real['date']), real['capacity_pred'],
             color=col, linewidth=2, alpha=0.9)
    if not fore.empty:
        ax1.plot(pd.to_datetime(fore['date']), fore['capacity_pred'],
                 color=col, linewidth=1.5, linestyle='--', alpha=0.55)
ax1.axhline(EOL_CAP, color='#f85149', linestyle='-.', linewidth=1.8, label='EOL 1.6 Ahr')
ax1.set_title('Predicted Capacity — All NASA Batteries | Solid=Real  Dashed=Forecast', fontsize=10, fontweight='bold')
ax1.set_ylabel('Capacity (Ahr)'); ax1.legend(fontsize=7)
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=7)

# Panel 2: SoH mini
ax2 = dark_ax(fig.add_subplot(gs[0, 2]))
for bat, df_b in all_results.items():
    ax2.plot(pd.to_datetime(df_b['date']), df_b['soh_pred_%'],
             color=colors_map[bat], linewidth=1.5, alpha=0.85)
ax2.axhline(80, color='#f85149', linestyle='-.', linewidth=1.5)
ax2.set_title('SoH (%) — 2 Years', fontweight='bold', fontsize=10)
ax2.set_ylabel('SoH (%)'); ax2.set_ylim([40, 115])

# Panel 3: Degradation Rate
ax3 = dark_ax(fig.add_subplot(gs[1, :2]))
for bat, df_b in all_results.items():
    ax3.plot(pd.to_datetime(df_b['date']),
             df_b['deg_rate_pred_smooth'] * 1000,
             color=colors_map[bat], linewidth=1.5, alpha=0.85)
ax3.axhline(0, color='white', linewidth=0.6, linestyle=':')
ax3.set_title('Degradation Rate (mAhr/cycle) — 7-cycle smooth', fontweight='bold', fontsize=10)
ax3.set_ylabel('mAhr/cycle')
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=7)

# Panel 4: KPI table
ax4 = dark_ax(fig.add_subplot(gs[1, 2]))
ax4.axis('off')
kpi_lines = ["  NASA 2-YEAR INFERENCE", "  " + "─" * 26]
for r in summary_rows:
    rul_str = f"{r['rul_pred_cycles_days']}d" if r['rul_pred_cycles_days'] < 9000 else ">2yr"
    kpi_lines.append(
        f"  {r['battery']}:  SoH={r['final_soh_pred_%']:.0f}%  "
        f"RUL={rul_str}  anomaly={r['anomaly_count']}"
    )
kpi_lines += [
    "", "  ─────────────────────────────",
    f"  Batteries  : {len(all_results)}",
    f"  Cycles/bat : {TARGET_CYCLES} (2 years)",
]
ax4.text(0.02, 0.98, "\n".join(kpi_lines),
         transform=ax4.transAxes, va='top', fontsize=8,
         color='#e6edf3', fontfamily='monospace',
         bbox=dict(boxstyle='round', fc='#161b22', ec='#388bfd', lw=1.5))

plt.suptitle('🔋 LSTM Battery Degradation — 2-Year NASA Data Inference Dashboard',
             fontsize=15, fontweight='bold', color='#e6edf3', y=1.01)
plt.savefig(os.path.join(PLOT_DIR, '31_nasa_2yr_dashboard.png'),
            dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("  → Saved: 31_nasa_2yr_dashboard.png")


# ─────────────────────────────────────────────
# FINAL PRINT
# ─────────────────────────────────────────────
print("\n" + "=" * 68)
print("  FINAL RESULTS — NASA DATA → 2-YEAR LSTM INFERENCE")
print("=" * 68)
print(f"\n  Batteries processed : {len(all_results)}")
print(f"  Total rows (all)    : {len(master_df)}")
print()
print(summary_df[['battery','real_cycles','forecast_cycles',
                  'final_soh_pred_%','rul_pred_cycles_days',
                  'eol_reached','mae_Ahr']].to_string(index=False))
print()
print("  Plots 24–31 saved → outputs/plots/")
print("  CSVs saved  → outputs/data/nasa_2yr_*.csv")
print("\n✓ Done!")
