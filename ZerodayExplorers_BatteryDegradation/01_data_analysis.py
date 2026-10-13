"""
01_data_analysis.py
===================
NASA Battery Dataset — Exploratory Data Analysis
Loads all .mat files, extracts discharge capacity curves,
and generates comprehensive visualisation plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.io as sio
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(DATA_DIR, 'outputs', 'plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

NOMINAL_CAPACITY = 2.0   # Ahr  (rated capacity for these cells)

BATTERY_FILES = [
    'B0025','B0026','B0027','B0028',
    'B0029','B0030','B0031','B0032',
    'B0033','B0034','B0036',
    'B0038','B0039','B0040',
    'B0041','B0042','B0043','B0044',
]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def load_mat(battery_id):
    path = os.path.join(DATA_DIR, f'{battery_id}.mat')
    if not os.path.exists(path):
        return None
    mat  = sio.loadmat(path, simplify_cells=True)
    return mat

def get_cycles(mat):
    """Return the 'cycle' struct array regardless of nesting."""
    # NASA mat files store: mat['B0025']['cycle'] or mat['cycle']
    key = [k for k in mat.keys() if not k.startswith('__')][0]
    battery = mat[key]
    if isinstance(battery, dict):
        return battery.get('cycle', [])
    return mat.get('cycle', [])

def extract_discharge_data(mat, battery_id):
    """
    Walk all cycles, pull discharge metrics per cycle.
    Returns DataFrame with columns:
      cycle_num, capacity, mean_voltage, mean_current,
      mean_temp, discharge_duration, ambient_temp
    """
    cycles      = get_cycles(mat)
    records     = []
    cycle_count = 0

    for cyc in cycles:
        try:
            ctype = str(cyc.get('type','')).strip().lower()
            if ctype != 'discharge':
                continue
            cycle_count += 1
            data = cyc.get('data', {})
            if not isinstance(data, dict):
                continue

            capacity = float(data.get('Capacity', np.nan))
            voltage  = np.atleast_1d(data.get('Voltage_measured', [np.nan]))
            current  = np.atleast_1d(data.get('Current_measured', [np.nan]))
            temp     = np.atleast_1d(data.get('Temperature_measured', [np.nan]))
            time_vec = np.atleast_1d(data.get('Time', [np.nan]))
            amb_temp = float(cyc.get('ambient_temperature', np.nan))

            records.append({
                'battery'           : battery_id,
                'cycle_num'         : cycle_count,
                'capacity'          : capacity,
                'mean_voltage'      : float(np.nanmean(voltage)),
                'min_voltage'       : float(np.nanmin(voltage)),
                'mean_current'      : float(np.nanmean(current)),
                'mean_temp'         : float(np.nanmean(temp)),
                'max_temp'          : float(np.nanmax(temp)),
                'discharge_duration': float(np.nanmax(time_vec) - np.nanmin(time_vec)),
                'ambient_temp'      : amb_temp,
            })
        except Exception:
            continue

    return pd.DataFrame(records)


def extract_impedance_data(mat, battery_id):
    """Extract Re and Rct from impedance cycles."""
    cycles  = get_cycles(mat)
    records = []
    imp_num = 0
    for cyc in cycles:
        try:
            ctype = str(cyc.get('type','')).strip().lower()
            if ctype != 'impedance':
                continue
            imp_num += 1
            data = cyc.get('data', {})
            if not isinstance(data, dict):
                continue
            Re  = float(np.nanmean(np.atleast_1d(data.get('Re',  [np.nan]))))
            Rct = float(np.nanmean(np.atleast_1d(data.get('Rct', [np.nan]))))
            records.append({'battery': battery_id, 'imp_num': imp_num, 'Re': Re, 'Rct': Rct})
        except Exception:
            continue
    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# LOAD ALL BATTERIES
# ─────────────────────────────────────────────
print("=" * 60)
print("NASA Battery Dataset — Data Analysis")
print("=" * 60)

all_discharge = []
all_impedance = []
summary_rows  = []

for bid in BATTERY_FILES:
    mat = load_mat(bid)
    if mat is None:
        print(f"  [SKIP] {bid}.mat not found")
        continue
    df_dis = extract_discharge_data(mat, bid)
    df_imp = extract_impedance_data(mat, bid)
    all_discharge.append(df_dis)
    all_impedance.append(df_imp)

    total_cycles = len(df_dis)
    if total_cycles > 0:
        cap_init = df_dis['capacity'].dropna().iloc[0]  if not df_dis['capacity'].dropna().empty else np.nan
        cap_end  = df_dis['capacity'].dropna().iloc[-1] if not df_dis['capacity'].dropna().empty else np.nan
        fade_pct = (1 - cap_end / cap_init) * 100 if (not np.isnan(cap_init) and cap_init > 0) else np.nan
        summary_rows.append({
            'battery'        : bid,
            'discharge_cycles': total_cycles,
            'impedance_cycles': len(df_imp),
            'cap_initial_Ahr' : round(cap_init, 4),
            'cap_final_Ahr'   : round(cap_end,  4),
            'total_fade_%'    : round(fade_pct,  2),
        })
    print(f"  [OK] {bid}: {total_cycles} discharge cycles, {len(df_imp)} impedance records")

df_all = pd.concat(all_discharge, ignore_index=True)
df_imp = pd.concat(all_impedance, ignore_index=True)
df_summary = pd.DataFrame(summary_rows)

print("\n── Dataset Summary ──────────────────────────────────")
print(df_summary.to_string(index=False))
print(f"\nTotal discharge records : {len(df_all)}")
print(f"Total impedance records : {len(df_imp)}")
print(f"Features available      : {list(df_all.columns)}")


# ─────────────────────────────────────────────
# PLOT 1: Capacity Fade — All Batteries
# ─────────────────────────────────────────────
print("\n[PLOT] 1 — Capacity fade curves ...")
batteries = df_all['battery'].unique()
colors     = cm.tab20(np.linspace(0, 1, len(batteries)))

fig, ax = plt.subplots(figsize=(14, 7))
for bat, col in zip(batteries, colors):
    sub = df_all[df_all['battery'] == bat].dropna(subset=['capacity'])
    ax.plot(sub['cycle_num'], sub['capacity'], marker='o', markersize=2.5,
            linewidth=1.5, label=bat, color=col)

ax.axhline(0.8 * NOMINAL_CAPACITY, color='red', linestyle='--', linewidth=1.5,
           label='80% SoH threshold (1.6 Ahr)')
ax.set_title('Battery Capacity Fade — All Batteries (NASA Dataset)', fontsize=14, fontweight='bold')
ax.set_xlabel('Discharge Cycle Number', fontsize=12)
ax.set_ylabel('Capacity (Ahr)', fontsize=12)
ax.legend(loc='upper right', fontsize=7, ncol=3)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_capacity_fade.png'), dpi=150)
plt.close()
print("  → Saved: outputs/plots/01_capacity_fade.png")


# ─────────────────────────────────────────────
# PLOT 2: State of Health (%) over cycles
# ─────────────────────────────────────────────
print("[PLOT] 2 — State of Health (%) ...")
fig, ax = plt.subplots(figsize=(14, 7))
for bat, col in zip(batteries, colors):
    sub = df_all[df_all['battery'] == bat].dropna(subset=['capacity'])
    if sub.empty: continue
    cap_init = sub['capacity'].iloc[0]
    if cap_init <= 0: continue
    soh = (sub['capacity'] / cap_init) * 100
    ax.plot(sub['cycle_num'], soh, marker='o', markersize=2.5, linewidth=1.5, label=bat, color=col)

ax.axhline(80, color='red', linestyle='--', linewidth=1.5, label='EOL = 80% SoH')
ax.set_title('State of Health (SoH %) — All Batteries', fontsize=14, fontweight='bold')
ax.set_xlabel('Discharge Cycle Number', fontsize=12)
ax.set_ylabel('SoH (%)', fontsize=12)
ax.legend(loc='upper right', fontsize=7, ncol=3)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_state_of_health.png'), dpi=150)
plt.close()
print("  → Saved: outputs/plots/02_state_of_health.png")


# ─────────────────────────────────────────────
# PLOT 3: Mean Discharge Voltage over cycles
# ─────────────────────────────────────────────
print("[PLOT] 3 — Mean discharge voltage over cycles ...")
fig, ax = plt.subplots(figsize=(14, 7))
for bat, col in zip(batteries, colors):
    sub = df_all[df_all['battery'] == bat].dropna(subset=['mean_voltage'])
    ax.plot(sub['cycle_num'], sub['mean_voltage'], linewidth=1.5, label=bat, color=col)

ax.set_title('Mean Discharge Voltage per Cycle', fontsize=14, fontweight='bold')
ax.set_xlabel('Discharge Cycle Number', fontsize=12)
ax.set_ylabel('Mean Voltage (V)', fontsize=12)
ax.legend(loc='lower left', fontsize=7, ncol=3)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_mean_voltage.png'), dpi=150)
plt.close()
print("  → Saved: outputs/plots/03_mean_voltage.png")


# ─────────────────────────────────────────────
# PLOT 4: Mean Temperature over cycles
# ─────────────────────────────────────────────
print("[PLOT] 4 — Mean discharge temperature over cycles ...")
fig, ax = plt.subplots(figsize=(14, 7))
for bat, col in zip(batteries, colors):
    sub = df_all[df_all['battery'] == bat].dropna(subset=['mean_temp'])
    ax.plot(sub['cycle_num'], sub['mean_temp'], linewidth=1.5, label=bat, color=col)

ax.set_title('Mean Discharge Temperature per Cycle', fontsize=14, fontweight='bold')
ax.set_xlabel('Discharge Cycle Number', fontsize=12)
ax.set_ylabel('Temperature (°C)', fontsize=12)
ax.legend(loc='upper left', fontsize=7, ncol=3)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_mean_temperature.png'), dpi=150)
plt.close()
print("  → Saved: outputs/plots/04_mean_temperature.png")


# ─────────────────────────────────────────────
# PLOT 5: Impedance Re and Rct over cycles
# ─────────────────────────────────────────────
print("[PLOT] 5 — Impedance (Re, Rct) over cycles ...")
if not df_imp.empty:
    imp_batteries = df_imp['battery'].unique()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for bat, col in zip(imp_batteries, cm.tab20(np.linspace(0, 1, len(imp_batteries)))):
        sub = df_imp[df_imp['battery'] == bat].dropna(subset=['Re'])
        if sub.empty: continue
        axes[0].plot(sub['imp_num'], sub['Re'],  linewidth=1.5, label=bat, color=col)
        sub2 = df_imp[df_imp['battery'] == bat].dropna(subset=['Rct'])
        if sub2.empty: continue
        axes[1].plot(sub2['imp_num'], sub2['Rct'], linewidth=1.5, label=bat, color=col)

    axes[0].set_title('Electrolyte Resistance (Re) Over Cycles', fontweight='bold')
    axes[0].set_xlabel('Impedance Measurement #')
    axes[0].set_ylabel('Re (Ω)')
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=0.3)

    axes[1].set_title('Charge Transfer Resistance (Rct) Over Cycles', fontweight='bold')
    axes[1].set_xlabel('Impedance Measurement #')
    axes[1].set_ylabel('Rct (Ω)')
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(alpha=0.3)

    plt.suptitle('EIS Impedance Parameters — Aging Indicators', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '05_impedance_Re_Rct.png'), dpi=150)
    plt.close()
    print("  → Saved: outputs/plots/05_impedance_Re_Rct.png")
else:
    print("  [SKIP] No impedance data found")


# ─────────────────────────────────────────────
# PLOT 6: Discharge Duration over cycles
# ─────────────────────────────────────────────
print("[PLOT] 6 — Discharge duration over cycles ...")
fig, ax = plt.subplots(figsize=(14, 7))
for bat, col in zip(batteries, colors):
    sub = df_all[df_all['battery'] == bat].dropna(subset=['discharge_duration'])
    ax.plot(sub['cycle_num'], sub['discharge_duration'], linewidth=1.5, label=bat, color=col)

ax.set_title('Discharge Duration per Cycle (seconds)', fontsize=14, fontweight='bold')
ax.set_xlabel('Discharge Cycle Number', fontsize=12)
ax.set_ylabel('Duration (s)', fontsize=12)
ax.legend(fontsize=7, ncol=3)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '06_discharge_duration.png'), dpi=150)
plt.close()
print("  → Saved: outputs/plots/06_discharge_duration.png")


# ─────────────────────────────────────────────
# PLOT 7: Feature Correlation Heatmap
# ─────────────────────────────────────────────
print("[PLOT] 7 — Feature correlation heatmap ...")
import seaborn as sns
feat_cols = ['capacity','mean_voltage','mean_current','mean_temp',
             'discharge_duration','min_voltage','max_temp']
df_corr = df_all[feat_cols].dropna()
corr = df_corr.corr()

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5, ax=ax)
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '07_feature_correlation.png'), dpi=150)
plt.close()
print("  → Saved: outputs/plots/07_feature_correlation.png")


# ─────────────────────────────────────────────
# PLOT 8: Degradation Rate (dCap/dCycle)
# ─────────────────────────────────────────────
print("[PLOT] 8 — Degradation rate (ΔCapacity/ΔCycle) ...")
fig, ax = plt.subplots(figsize=(14, 7))
for bat, col in zip(batteries, colors):
    sub = df_all[df_all['battery'] == bat].dropna(subset=['capacity']).copy()
    if len(sub) < 3: continue
    sub['deg_rate'] = sub['capacity'].diff() / sub['cycle_num'].diff()
    sub_smooth = sub['deg_rate'].rolling(3, center=True).mean()
    ax.plot(sub['cycle_num'], sub_smooth, linewidth=1.5, label=bat, color=col)

ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_title('Degradation Rate (ΔCapacity/ΔCycle) — Rolling Mean(3)', fontsize=14, fontweight='bold')
ax.set_xlabel('Discharge Cycle Number', fontsize=12)
ax.set_ylabel('ΔAhr / Cycle', fontsize=12)
ax.legend(fontsize=7, ncol=3)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '08_degradation_rate.png'), dpi=150)
plt.close()
print("  → Saved: outputs/plots/08_degradation_rate.png")


# ─────────────────────────────────────────────
# SAVE: Summary CSV
# ─────────────────────────────────────────────
csv_dir = os.path.join(DATA_DIR, 'outputs', 'data')
os.makedirs(csv_dir, exist_ok=True)
df_summary.to_csv(os.path.join(csv_dir, 'battery_summary_eda.csv'), index=False)
df_all.to_csv(os.path.join(csv_dir, 'all_discharge_cycles.csv'), index=False)
df_imp.to_csv(os.path.join(csv_dir, 'all_impedance_cycles.csv'), index=False)
print("\n[CSV] outputs/data/battery_summary_eda.csv")
print("[CSV] outputs/data/all_discharge_cycles.csv")
print("[CSV] outputs/data/all_impedance_cycles.csv")

print("\n✓ EDA complete — 8 plots + 3 CSVs saved to outputs/")
