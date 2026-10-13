"""
train.py
--------
Main training & evaluation script for the physics-constrained CNN-LSTM
energy forecaster.  Calibrated for Gwalior, MP microgrid.

Key fixes vs v1:
  - PHYSICS_PENALTY reduced from 10.0 to 0.1 (v1 caused model collapse)
  - Softplus replaced by quantile sorting in model (fixes uncertainty = 0)
  - 20-epoch warm-up: first WARMUP_EPOCHS use pinball-only loss so the model
    learns the forecasting signal BEFORE physics constraints are applied.
    After warm-up, physics penalty ramps in linearly over 10 epochs.

Usage:
    python train.py
"""

# -- Standard library -----------------------------------------------------------
import os
import copy
import time

# -- Third-party ----------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# -- Local modules --------------------------------------------------------------
from config import (
    DATA_CSV, MODEL_SAVE_PATH, EPOCHS, PATIENCE, WARMUP_EPOCHS,
    LEARNING_RATE, BATCH_SIZE, HORIZON, QUANTILES, SEED,
    IDX_SOLAR, IDX_WIND, IDX_LOAD, Q_LOW, Q_MID, Q_HIGH,
    PHYSICS_PENALTY,
)
from dataset import get_dataloaders, ALL_FEATURES, TARGET_COLS
from model   import CNNLSTMForecaster
from losses  import PhysicsConstrainedQuantileLoss, compute_uncertainty
from utils   import (
    plot_loss_curve, plot_forecast,
    print_uncertainty_table, print_epoch_summary,
    ensure_output_dir,
)


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*65}")
print(f"  CNN-LSTM Gwalior Microgrid Forecaster  (v2 — fixed)")
print(f"{'='*65}")
print(f"  Device         : {DEVICE}")
print(f"  Targets        : {TARGET_COLS}")
print(f"  Physics penalty: {PHYSICS_PENALTY} (was 10.0 — caused collapse)")
print(f"  Warm-up epochs : {WARMUP_EPOCHS} (pinball only, no physics penalty)")
print(f"{'='*65}\n")


# -----------------------------------------------------------------------------
# Step 1 – Data
# -----------------------------------------------------------------------------
if not os.path.exists(DATA_CSV):
    print("[train] Data CSV not found. Running fetch_data.py ...")
    from fetch_data import main as fetch_main
    fetch_main()
else:
    print(f"[train] Using dataset: {DATA_CSV}")

(train_loader, val_loader, test_loader,
 feat_scaler, tgt_scaler,
 ghi_raw_idx, wind_raw_idx) = get_dataloaders()


# -----------------------------------------------------------------------------
# Step 2 – Model, Loss, Optimizer
# -----------------------------------------------------------------------------
n_features = len(ALL_FEATURES)
model      = CNNLSTMForecaster(n_features=n_features).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n[model] Parameters   : {n_params:,}")
print(f"[model] Feature dim  : {n_features}")
print(f"[model] Horizon      : {HORIZON} h\n")

violation_counter: list[int] = []
criterion = PhysicsConstrainedQuantileLoss(
    tgt_scaler=tgt_scaler,
    violation_counter=violation_counter,
)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

# Physics ramp: weight goes from 0 -> 1 over RAMP_EPOCHS after warm-up
RAMP_EPOCHS = 10


def physics_weight(epoch: int) -> float:
    """
    Warm-up schedule:
      epoch <= WARMUP_EPOCHS       : 0.0  (pinball only)
      WARMUP_EPOCHS < epoch <= end : ramp linearly from 0 -> 1 over RAMP_EPOCHS
      thereafter                  : 1.0  (full physics penalty)
    """
    if epoch <= WARMUP_EPOCHS:
        return 0.0
    progress = (epoch - WARMUP_EPOCHS) / RAMP_EPOCHS
    return min(progress, 1.0)


# -----------------------------------------------------------------------------
# Helper: Pure pinball loss (no physics) — used during warm-up
# -----------------------------------------------------------------------------
def pinball_only(pred_norm: torch.Tensor,
                 y_norm   : torch.Tensor) -> torch.Tensor:
    """
    Standalone pinball loss without any physics penalty.
    Used during warm-up epochs so the model first learns the
    forecasting signal before physics constraints kick in.
    """
    q   = torch.tensor(QUANTILES, dtype=torch.float32, device=pred_norm.device)
    y   = y_norm.unsqueeze(-1)                      # (B, H, 3, 1)
    res = y - pred_norm                             # (B, H, 3, Q)
    return torch.max(q * res, (q - 1) * res).mean()


# -----------------------------------------------------------------------------
# Helper: one epoch pass
# -----------------------------------------------------------------------------
def run_epoch(loader, is_train: bool, epoch: int) -> float:
    model.train(is_train)
    total_loss = 0.0
    n_batches  = 0
    pw = physics_weight(epoch)

    with torch.set_grad_enabled(is_train):
        for x_norm, y_norm, x_raw, y_raw in loader:
            x_norm = x_norm.to(DEVICE)
            y_norm = y_norm.to(DEVICE)
            x_raw  = x_raw .to(DEVICE)

            pred_norm = model(x_norm)               # (B, H, 3, Q)

            if pw == 0.0:
                # Warm-up: pinball only
                loss = pinball_only(pred_norm, y_norm)
            else:
                # Full / ramping physics loss
                pb_loss = criterion._pinball(pred_norm, y_norm)
                s_pen, _ = criterion._solar_penalty(pred_norm, x_raw)
                b_pen, _ = criterion._betz_penalty (pred_norm, x_raw)
                f_pen, _ = criterion._load_floor_penalty(pred_norm)
                loss = pb_loss + pw * (s_pen + b_pen + f_pen)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

    return total_loss / max(n_batches, 1)


# -----------------------------------------------------------------------------
# Step 3 – Training loop
# -----------------------------------------------------------------------------
ensure_output_dir()
train_losses: list[float] = []
val_losses  : list[float] = []

best_val_loss     = float("inf")
best_model_wts    = copy.deepcopy(model.state_dict())
epochs_no_improve = 0

print("-" * 65)
print(f"  Training: {EPOCHS} epochs | patience={PATIENCE} | "
      f"warm-up={WARMUP_EPOCHS}")
print("-" * 65)

for epoch in range(1, EPOCHS + 1):
    pw = physics_weight(epoch)
    violations_this_epoch = []
    criterion.violation_counter = violations_this_epoch

    train_loss = run_epoch(train_loader, is_train=True,  epoch=epoch)
    val_loss   = run_epoch(val_loader,   is_train=False, epoch=epoch)

    train_losses.append(train_loss)
    val_losses  .append(val_loss)

    total_viol = sum(violations_this_epoch) if pw > 0 else 0

    # Phase label
    if epoch <= WARMUP_EPOCHS:
        phase = f"[WARMUP  phys_w=0.00]"
    elif pw < 1.0:
        phase = f"[RAMP    phys_w={pw:.2f}]"
    else:
        phase = f"[PHYSICS phys_w=1.00]"

    print(f"Epoch {epoch:>3} {phase} Train:{train_loss:.4f} "
          f"Val:{val_loss:.4f} Viol:{total_viol:,}")

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss  = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        torch.save({
            "epoch"      : epoch,
            "model_state": model.state_dict(),
            "optimizer"  : optimizer.state_dict(),
            "val_loss"   : best_val_loss,
            "n_features" : n_features,
            "feat_scaler": feat_scaler,
            "tgt_scaler" : tgt_scaler,
        }, MODEL_SAVE_PATH)
        print(f"         ** Saved best model (val={best_val_loss:.5f}) **")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"\n[train] Early stopping at epoch {epoch} "
                  f"({PATIENCE} epochs without improvement)")
            break

print(f"\n[train] Best val loss : {best_val_loss:.6f}")
print(f"[train] Model saved   : {MODEL_SAVE_PATH}")

model.load_state_dict(best_model_wts)


# -----------------------------------------------------------------------------
# Step 4 – Plots & uncertainty table
# -----------------------------------------------------------------------------
plot_loss_curve(train_losses, val_losses)

model.eval()
x_norm_t, y_norm_t, x_raw_t, y_raw_t = next(iter(test_loader))
x_norm_t = x_norm_t[:1].to(DEVICE)
y_raw_t  = y_raw_t [:1]

with torch.no_grad():
    pred_norm_t = model(x_norm_t)

tgt_mean  = torch.tensor(tgt_scaler.mean_,  dtype=torch.float32)
tgt_scale = torch.tensor(tgt_scaler.scale_, dtype=torch.float32)

pred_raw_t  = (pred_norm_t.cpu()
               * tgt_scale[None, None, :, None]
               + tgt_mean [None, None, :, None])

pred_raw_np = pred_raw_t.squeeze(0).numpy()   # (H, 3, Q)
y_raw_np    = y_raw_t   .squeeze(0).numpy()   # (H, 3)

pred_raw_tensor = torch.tensor(pred_raw_np)
uncertainty_np  = compute_uncertainty(pred_raw_tensor).numpy()

print_uncertainty_table(uncertainty_np)
plot_forecast(pred_raw_np, y_raw_np, uncertainty_np)

print("\n" + "=" * 65)
print("  TRAINING COMPLETE")
print(f"  Model   : {MODEL_SAVE_PATH}")
print(f"  Loss    : outputs/loss_curve.png")
print(f"  Forecast: outputs/sample_forecast.png")
print("=" * 65 + "\n")
print("  Now run: python quick_eval.py")
