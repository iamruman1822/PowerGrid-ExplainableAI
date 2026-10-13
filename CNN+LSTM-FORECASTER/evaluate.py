"""
evaluate.py
-----------
Comprehensive accuracy evaluation + inference report for the
CNN-LSTM Gwalior microgrid forecaster.

Metrics computed on the held-out TEST set (last 15% of data):
  - MAE  (Mean Absolute Error)         — lower is better
  - RMSE (Root Mean Squared Error)     — lower is better
  - MAPE (Mean Absolute % Error)       — lower is better
  - Coverage (% of actuals inside 10th-90th PI band) — target >= 80%
  - Pinball loss at each quantile      — lower is better
  - Physics violation rate (%)         — target = 0%

Then runs a sample 24-h inference and prints full timestep table.
"""

import os
import sys
import numpy as np
import torch

# -----------------------------------------------------------------------------
print("=" * 65)
print("  CNN-LSTM GWALIOR MICROGRID — EVALUATION REPORT")
print("=" * 65)

# -- Load dataset & model -----------------------------------------------------
print("\n[1] Loading checkpoint and test data ...")
from dataset import get_dataloaders, ALL_FEATURES, TARGET_COLS
from model   import CNNLSTMForecaster
from losses  import compute_uncertainty, PhysicsConstrainedQuantileLoss
from config  import (
    MODEL_SAVE_PATH, HORIZON, QUANTILES,
    IDX_SOLAR, IDX_WIND, IDX_LOAD,
    Q_LOW, Q_MID, Q_HIGH,
    BETZ_COEFF, AIR_DENSITY, ROTOR_AREA,
    SOLAR_PANEL_AREA, SOLAR_PANEL_EFF, SOLAR_TEMP_COEFF, SOLAR_NOCT_TEMP,
    WIND_CUT_IN, WIND_RATED, WIND_CUT_OUT, LOAD_MIN_KW,
)
from dataset import WIND_SPEED_RAW_IDX, GHI_RAW_IDX, TEMP_RAW_IDX

(train_loader, val_loader, test_loader,
 feat_scaler, tgt_scaler, _, _) = get_dataloaders()

ckpt = torch.load(MODEL_SAVE_PATH, map_location="cpu", weights_only=False)
model = CNNLSTMForecaster(n_features=len(ALL_FEATURES))
model.load_state_dict(ckpt["model_state"])
model.eval()

best_epoch    = ckpt.get("epoch", "?")
best_val_loss = ckpt.get("val_loss", float("nan"))
print(f"    Checkpoint  : epoch {best_epoch}, best val loss = {best_val_loss:.5f}")
print(f"    Test batches: {len(test_loader)}")

tgt_mean  = torch.tensor(tgt_scaler.mean_,  dtype=torch.float32)
tgt_scale = torch.tensor(tgt_scaler.scale_, dtype=torch.float32)
NOCT = 45.0


def inverse(pred_norm):
    """(B, H, 3, Q) normalised -> (B, H, 3, Q) raw kWh"""
    return pred_norm * tgt_scale[None, None, :, None] + tgt_mean[None, None, :, None]


# -----------------------------------------------------------------------------
# [2] Full test-set evaluation
# -----------------------------------------------------------------------------
print("\n[2] Running inference on full TEST set ...")

all_pred_raw = []   # list of (B, H, 3, Q)
all_y_raw    = []   # list of (B, H, 3)
all_x_raw    = []   # list of (B, SEQ, F)

with torch.no_grad():
    for x_norm, y_norm, x_raw, y_raw in test_loader:
        pred_norm = model(x_norm)
        pred_raw  = inverse(pred_norm)
        all_pred_raw.append(pred_raw.numpy())
        all_y_raw   .append(y_raw.numpy())
        all_x_raw   .append(x_raw.numpy())

pred_all = np.concatenate(all_pred_raw, axis=0)   # (N, H, 3, 3)
y_all    = np.concatenate(all_y_raw,    axis=0)   # (N, H, 3)
x_all    = np.concatenate(all_x_raw,    axis=0)   # (N, SEQ, F)

N = len(pred_all)
print(f"    Test samples: {N:,}  x  {HORIZON} h  x  3 targets")


# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------
def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))

def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat)**2)))

def mape(y, yhat, eps=1.0):
    mask = np.abs(y) > eps
    return float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100)

def coverage(y, lo, hi):
    """% of actuals inside [lo, hi] prediction interval"""
    inside = ((y >= lo) & (y <= hi))
    return float(inside.mean() * 100)

def pinball(y, yhat, q):
    r = y - yhat
    return float(np.mean(np.where(r >= 0, q * r, (q - 1) * r)))


# -----------------------------------------------------------------------------
# [3] Per-target metrics
# -----------------------------------------------------------------------------
print("\n[3] Computing accuracy metrics ...")

target_names     = ["Solar Power", "Wind Power ", "Load Demand"]
target_units     = ["kWh",         "kWh",         "kW"]

sep = "-" * 75
print(f"\n{sep}")
print(f"  {'TARGET':<14} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8} "
      f"{'Coverage':>10} {'PB_10':>8} {'PB_50':>8} {'PB_90':>8}")
print(sep)

results = {}
for i, (name, unit) in enumerate(zip(target_names, target_units)):
    y    = y_all   [:, :, i].flatten()        # actual
    lo   = pred_all[:, :, i, Q_LOW ].flatten()
    mid  = pred_all[:, :, i, Q_MID ].flatten()
    hi   = pred_all[:, :, i, Q_HIGH].flatten()

    mae_v  = mae(y, mid)
    rmse_v = rmse(y, mid)
    mape_v = mape(y, mid)
    cov_v  = coverage(y, lo, hi)
    pb10   = pinball(y, lo,  QUANTILES[0])
    pb50   = pinball(y, mid, QUANTILES[1])
    pb90   = pinball(y, hi,  QUANTILES[2])

    results[name] = dict(MAE=mae_v, RMSE=rmse_v, MAPE=mape_v,
                         Coverage=cov_v, PB10=pb10, PB50=pb50, PB90=pb90,
                         unit=unit)

    cov_flag = "OK" if cov_v >= 80 else "LOW"
    print(f"  {name:<14} {mae_v:>8.3f} {rmse_v:>8.3f} {mape_v:>7.2f}%"
          f" {cov_v:>8.1f}% {cov_flag:<3}"
          f" {pb10:>8.3f} {pb50:>8.3f} {pb90:>8.3f}")

print(sep)
print(f"  Units: MAE/RMSE in kWh (kW for load) | Coverage: % actuals inside 10-90% PI")


# -----------------------------------------------------------------------------
# [4] Physics violation rate on test set
# -----------------------------------------------------------------------------
print(f"\n[4] Physics constraint violation rate ...")

# Solar violations
ghi_last  = x_all[:, -1, GHI_RAW_IDX]    # (N,)
temp_last = x_all[:, -1, TEMP_RAW_IDX]   # (N,)
t_cell    = temp_last + (NOCT - 20.0) / 800.0 * ghi_last
eff_adj   = np.clip(
    SOLAR_PANEL_EFF * (1.0 + SOLAR_TEMP_COEFF * (t_cell - SOLAR_NOCT_TEMP)), 0.05, None)
solar_ceil = (eff_adj * SOLAR_PANEL_AREA * ghi_last / 1000.0)[:, None, None]  # (N,1,1)
solar_pred = pred_all[:, :, IDX_SOLAR, :]  # (N, H, Q)
n_solar_viol = int(np.sum(solar_pred > solar_ceil))
total_solar  = solar_pred.size

# Wind violations
v_last    = x_all[:, -1, WIND_SPEED_RAW_IDX]   # (N,)
p_aero    = BETZ_COEFF * 0.5 * AIR_DENSITY * ROTOR_AREA * v_last**3 / 1000.0
p_rated   = BETZ_COEFF * 0.5 * AIR_DENSITY * ROTOR_AREA * WIND_RATED**3 / 1000.0
wind_ceil = np.where(v_last < WIND_CUT_IN,  0.0,
            np.where(v_last < WIND_RATED,   p_aero,
            np.where(v_last < WIND_CUT_OUT, p_rated, 0.0)))[:, None, None]  # (N,1,1)
wind_pred = pred_all[:, :, IDX_WIND, :]  # (N, H, Q)
n_wind_viol  = int(np.sum(wind_pred > wind_ceil))
total_wind   = wind_pred.size

# Load floor violations
load_pred    = pred_all[:, :, IDX_LOAD, :]
n_load_viol  = int(np.sum(load_pred < LOAD_MIN_KW))
total_load   = load_pred.size

print(f"  Solar ceiling violations : {n_solar_viol:>8,} / {total_solar:,}  "
      f"({100*n_solar_viol/total_solar:.3f}%)")
print(f"  Wind Betz violations     : {n_wind_viol:>8,} / {total_wind:,}  "
      f"({100*n_wind_viol/total_wind:.3f}%)")
print(f"  Load floor violations    : {n_load_viol:>8,} / {total_load:,}  "
      f"({100*n_load_viol/total_load:.3f}%)")


# -----------------------------------------------------------------------------
# [5] Inference: first test sample as a 24-h forecast
# -----------------------------------------------------------------------------
print(f"\n[5] Sample 24-h Inference (first test window) ...")
print(sep)

sample_pred = pred_all[0]    # (H, 3, Q)
sample_y    = y_all   [0]    # (H, 3)
pred_tensor = torch.tensor(sample_pred)
uncertainty = compute_uncertainty(pred_tensor).numpy()  # (H, 3)

hdr = (f"  {'Hr':>3}  "
       f"{'Solar_Lo':>9} {'Solar_Mid':>10} {'Solar_Hi':>9} {'Solar_Act':>10} {'S_Unc':>6}  "
       f"{'Wind_Mid':>9} {'Wind_Act':>9} {'W_Unc':>6}  "
       f"{'Load_Mid':>9} {'Load_Act':>9} {'L_Unc':>6}")
print(hdr)
print(sep)
for h in range(HORIZON):
    sl, sm, sh = sample_pred[h, IDX_SOLAR, Q_LOW], sample_pred[h, IDX_SOLAR, Q_MID], sample_pred[h, IDX_SOLAR, Q_HIGH]
    wm         = sample_pred[h, IDX_WIND, Q_MID]
    lm         = sample_pred[h, IDX_LOAD, Q_MID]
    sa, wa, la = sample_y[h, IDX_SOLAR], sample_y[h, IDX_WIND], sample_y[h, IDX_LOAD]
    su, wu, lu = uncertainty[h, IDX_SOLAR], uncertainty[h, IDX_WIND], uncertainty[h, IDX_LOAD]

    print(f"  {h+1:>3}  "
          f"{sl:>9.3f} {sm:>10.3f} {sh:>9.3f} {sa:>10.3f} {su:>6.3f}  "
          f"{wm:>9.3f} {wa:>9.3f} {wu:>6.3f}  "
          f"{lm:>9.1f} {la:>9.1f} {lu:>6.3f}")

print(sep)


# -----------------------------------------------------------------------------
# [6] Summary report
# -----------------------------------------------------------------------------
print(f"\n{'='*65}")
print("  SUMMARY REPORT")
print(f"{'='*65}")
print(f"  Location    : Gwalior, Lakhnotikhurd, MP (26.193N, 78.1487E)")
print(f"  Dataset     : Open-Meteo real weather 2023-2024 (17,520 rows)")
print(f"  Test set    : last 15% = {N:,} windows of {HORIZON}h")
print(f"  Best epoch  : {best_epoch}")
print(f"  Best val loss: {best_val_loss:.5f}")
print()
for name, r in results.items():
    grade = ("GOOD" if r["MAPE"] < 20 else
             "FAIR" if r["MAPE"] < 40 else "NEEDS TUNING")
    print(f"  {name}")
    print(f"    MAE={r['MAE']:.3f} {r['unit']}  RMSE={r['RMSE']:.3f}  "
          f"MAPE={r['MAPE']:.2f}%  Coverage={r['Coverage']:.1f}%  [{grade}]")
print()

total_viol_pct = 100 * (n_solar_viol + n_wind_viol + n_load_viol) / (
    total_solar + total_wind + total_load)
print(f"  Physics violations: {total_viol_pct:.4f}% of all predictions")
if total_viol_pct < 1.0:
    print("  Physics compliance: GOOD (< 1% violations)")
else:
    print("  Physics compliance: NEEDS TUNING (reduce PHYSICS_PENALTY?)")

print(f"\n  Plots saved to outputs/")
print(f"  Model: {MODEL_SAVE_PATH}")
print(f"{'='*65}\n")
