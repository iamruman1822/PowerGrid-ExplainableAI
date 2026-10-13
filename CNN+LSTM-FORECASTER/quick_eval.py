"""Quick evaluation — pure ASCII output, no special characters."""
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

from dataset import get_dataloaders, ALL_FEATURES, TARGET_COLS
from model   import CNNLSTMForecaster
from losses  import compute_uncertainty
from config  import (
    MODEL_SAVE_PATH, HORIZON, QUANTILES,
    Q_LOW, Q_MID, Q_HIGH,
    IDX_SOLAR, IDX_WIND, IDX_LOAD,
    BETZ_COEFF, AIR_DENSITY, ROTOR_AREA,
    WIND_CUT_IN, WIND_RATED, WIND_CUT_OUT,
    SOLAR_PANEL_AREA, SOLAR_PANEL_EFF,
    SOLAR_TEMP_COEFF, SOLAR_NOCT_TEMP,
    LOAD_MIN_KW,
)
from dataset import WIND_SPEED_RAW_IDX, GHI_RAW_IDX, TEMP_RAW_IDX

NOCT = 45.0

# --- Load model & data ---
(_, _, test_loader, fs, ts, _, _) = get_dataloaders()
ckpt  = torch.load(MODEL_SAVE_PATH, map_location="cpu", weights_only=False)
model = CNNLSTMForecaster(n_features=len(ALL_FEATURES))
model.load_state_dict(ckpt["model_state"])
model.eval()

tm  = torch.tensor(ts.mean_,  dtype=torch.float32)
tsc = torch.tensor(ts.scale_, dtype=torch.float32)

def inv(p):
    return p * tsc[None, None, :, None] + tm[None, None, :, None]

preds, actuals, xs = [], [], []
with torch.no_grad():
    for xn, yn, xr, yr in test_loader:
        preds.append(inv(model(xn)).numpy())
        actuals.append(yr.numpy())
        xs.append(xr.numpy())

P = np.concatenate(preds)    # (N, H, 3, 3)
Y = np.concatenate(actuals)  # (N, H, 3)
X = np.concatenate(xs)       # (N, SEQ, F)
N = len(P)

# --- Metrics ---
def mae(y, h):  return float(np.mean(np.abs(y - h)))
def rmse(y, h): return float(np.sqrt(np.mean((y - h)**2)))
def mape(y, h):
    m = np.abs(y) > 1.0
    return float(np.mean(np.abs((y[m] - h[m]) / y[m])) * 100)
def cov(y, lo, hi): return float(np.mean((y >= lo) & (y <= hi)) * 100)

names = ["Solar (kWh)", "Wind  (kWh)", "Load  (kW) "]

print("=" * 65)
print("  CNN-LSTM GWALIOR MICROGRID - EVALUATION REPORT")
print("=" * 65)
print(f"  Data     : Open-Meteo real weather 2023-2024 (17,520 h)")
print(f"  Test set : last 15% = {N:,} windows x {HORIZON}h x 3 targets")
print(f"  Best epoch  : {ckpt['epoch']}")
print(f"  Best val loss: {ckpt['val_loss']:.5f}")
print()

print(f"  {'TARGET':<14} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8} {'PI Cov%':>9}  Grade")
print("  " + "-" * 59)
grades = []
for i, name in enumerate(names):
    y   = Y[:, :, i].flatten()
    lo  = P[:, :, i, Q_LOW].flatten()
    mid = P[:, :, i, Q_MID].flatten()
    hi  = P[:, :, i, Q_HIGH].flatten()
    m   = mae(y, mid)
    r   = rmse(y, mid)
    mp  = mape(y, mid)
    c   = cov(y, lo, hi)
    grade = "GOOD" if mp < 20 else ("FAIR" if mp < 40 else "NEEDS TUNING")
    grades.append(grade)
    print(f"  {name:<14} {m:>8.3f} {r:>8.3f} {mp:>7.2f}% {c:>8.1f}%  {grade}")

print()
print("  PI Cov% = % of actual values inside [10th, 90th] prediction band")
print("  Target: Coverage >= 80%, MAPE < 20% = GOOD")

# --- Physics violations ---
print()
print("  PHYSICS CONSTRAINT VIOLATIONS (Test Set):")
print("  " + "-" * 50)

ghi  = X[:, -1, GHI_RAW_IDX]
temp = X[:, -1, TEMP_RAW_IDX]
t_c  = temp + (NOCT - 20.0) / 800.0 * ghi
eff  = np.clip(SOLAR_PANEL_EFF * (1 + SOLAR_TEMP_COEFF * (t_c - SOLAR_NOCT_TEMP)), 0.05, None)
s_ceil = (eff * SOLAR_PANEL_AREA * ghi / 1000.0)[:, None, None]
sv = int(np.sum(P[:, :, IDX_SOLAR, :] > s_ceil))
st = P[:, :, IDX_SOLAR, :].size

v    = X[:, -1, WIND_SPEED_RAW_IDX]
pa   = BETZ_COEFF * 0.5 * AIR_DENSITY * ROTOR_AREA * v**3 / 1000.0
pr   = BETZ_COEFF * 0.5 * AIR_DENSITY * ROTOR_AREA * WIND_RATED**3 / 1000.0
w_c  = np.where(v < WIND_CUT_IN, 0.0,
       np.where(v < WIND_RATED, pa,
       np.where(v < WIND_CUT_OUT, pr, 0.0)))[:, None, None]
wv   = int(np.sum(P[:, :, IDX_WIND, :] > w_c))
wt   = P[:, :, IDX_WIND, :].size

lv   = int(np.sum(P[:, :, IDX_LOAD, :] < LOAD_MIN_KW))
lt   = P[:, :, IDX_LOAD, :].size

print(f"  Solar ceiling  : {sv:>8,} / {st:,}  ({100*sv/st:.3f}%)")
print(f"  Wind (IEC 61400): {wv:>7,} / {wt:,}  ({100*wv/wt:.3f}%)")
print(f"  Load floor     : {lv:>8,} / {lt:,}  ({100*lv/lt:.3f}%)")

total_viol_pct = 100 * (sv + wv + lv) / (st + wt + lt)
print(f"  Overall physics: {total_viol_pct:.4f}%  -> {'COMPLIANT (< 1%)' if total_viol_pct < 1 else 'NEEDS TUNING'}")

# --- 24h sample inference ---
print()
print("  24-HOUR SAMPLE INFERENCE (first test window):")
print("  " + "-" * 68)
print(f"  {'Hr':>3}  {'Solar_Pred':>10} {'Solar_Act':>10} {'S_Unc':>6}  "
      f"{'Wind_Pred':>9} {'Wind_Act':>8} {'W_Unc':>6}  "
      f"{'Load_Pred':>9} {'Load_Act':>9} {'L_Unc':>6}")
print("  " + "-" * 68)

sp  = P[0]     # (H, 3, 3)
sy  = Y[0]     # (H, 3)
unc = compute_uncertainty(torch.tensor(sp)).numpy()  # (H, 3)

for h in range(HORIZON):
    sm = sp[h, IDX_SOLAR, Q_MID];  sa = sy[h, IDX_SOLAR]; su = unc[h, IDX_SOLAR]
    wm = sp[h, IDX_WIND,  Q_MID];  wa = sy[h, IDX_WIND];  wu = unc[h, IDX_WIND]
    lm = sp[h, IDX_LOAD,  Q_MID];  la = sy[h, IDX_LOAD];  lu = unc[h, IDX_LOAD]
    print(f"  +{h+1:02d}h  {sm:>10.3f} {sa:>10.3f} {su:>6.3f}  "
          f"{wm:>9.3f} {wa:>8.3f} {wu:>6.3f}  "
          f"{lm:>9.1f} {la:>9.1f} {lu:>6.3f}")

print("  " + "-" * 68)
print()
print("  Unc = (90th-10th)/median, clipped [0,1]. Lower = more confident.")
print()
print("=" * 65)
print("  DONE  |  Model: outputs/cnn_lstm_forecaster.pt")
print("=" * 65)
