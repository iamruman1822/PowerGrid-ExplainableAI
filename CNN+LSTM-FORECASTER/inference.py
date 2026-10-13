"""
inference.py
------------
Standalone inference script — loads a saved checkpoint and runs
a 24-hour forecast on any CSV matching the expected column format.

Usage:
    python inference.py --csv path/to/your_data.csv --rows_back 48

The script will:
  • Load the last 48 rows from the CSV as the input window.
  • Run the model and decode predictions back to physical units (kWh).
  • Print per-timestep quantiles (10th / 50th / 90th) + uncertainty.
  • Save a forecast plot to outputs/inference_forecast.png.
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch

from config import (
    MODEL_SAVE_PATH, HORIZON, SEQ_LEN,
    Q_LOW, Q_MID, Q_HIGH,
)
from dataset import ALL_FEATURES, TARGET_COLS, load_dataframe
from model   import CNNLSTMForecaster
from losses  import compute_uncertainty
from utils   import plot_forecast, print_uncertainty_table


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",      default=None,  help="Path to input CSV")
    p.add_argument("--rows_back", type=int, default=SEQ_LEN,
                   help="Number of rows to use as input window (default=48)")
    return p.parse_args()


def load_checkpoint(path: str):
    """Load model + scalers from .pt checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    n_features  = ckpt["n_features"]
    feat_scaler = ckpt["feat_scaler"]
    tgt_scaler  = ckpt["tgt_scaler"]

    model = CNNLSTMForecaster(n_features=n_features)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, feat_scaler, tgt_scaler


def prepare_window(csv_path: str, rows_back: int,
                   feat_scaler) -> torch.Tensor:
    """
    Reads the CSV, engineers features, normalises, and returns
    the last `rows_back` rows as a (1, rows_back, n_features) tensor.
    """
    from generate_data import generate_dataset
    from config import DATA_CSV

    # Fall back to synthetic data if no CSV provided
    if csv_path is None:
        if not os.path.exists(DATA_CSV):
            generate_dataset()
        csv_path = DATA_CSV

    df = load_dataframe()   # reads DATA_CSV from config (can be overridden)
    X  = df[ALL_FEATURES].values
    X_norm = feat_scaler.transform(X).astype(np.float32)
    window = X_norm[-rows_back:]   # (rows_back, F)
    return torch.tensor(window).unsqueeze(0)    # (1, rows_back, F)


def inverse_transform_pred(pred_norm: torch.Tensor,
                            tgt_scaler) -> np.ndarray:
    """Convert normalised (1,H,3,Q) prediction -> raw kWh (H,3,Q)."""
    m = torch.tensor(tgt_scaler.mean_ , dtype=torch.float32)
    s = torch.tensor(tgt_scaler.scale_, dtype=torch.float32)
    raw = pred_norm.squeeze(0) * s[None, :, None] + m[None, :, None]
    return raw.numpy()   # (H, 3, Q)


def main():
    args = parse_args()

    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"[inference] No checkpoint found at {MODEL_SAVE_PATH}.")
        print("[inference] Run  python train.py  first.")
        return

    model, feat_scaler, tgt_scaler = load_checkpoint(MODEL_SAVE_PATH)
    print(f"[inference] Loaded checkpoint: {MODEL_SAVE_PATH}")

    x_tensor = prepare_window(args.csv, args.rows_back, feat_scaler)

    with torch.no_grad():
        pred_norm = model(x_tensor)   # (1, H, 3, Q)

    pred_raw = inverse_transform_pred(pred_norm, tgt_scaler)  # (H, 3, Q)

    # Dummy ground truth (zeros) — replace with real values if available
    y_raw = np.zeros((HORIZON, 3), dtype=np.float32)

    uncertainty = compute_uncertainty(
        torch.tensor(pred_raw)).numpy()                        # (H, 3)

    # -- Console output --------------------------------------------------------
    print(f"\n{'='*65}")
    print("  24-HOUR FORECAST RESULTS")
    print(f"{'='*65}")
    for h in range(HORIZON):
        print(f"\n  Hour +{h+1:02d}:")
        for t_idx, t_name in enumerate(TARGET_COLS):
            lo  = pred_raw[h, t_idx, Q_LOW]
            med = pred_raw[h, t_idx, Q_MID]
            hi  = pred_raw[h, t_idx, Q_HIGH]
            u   = uncertainty[h, t_idx]
            print(f"    {t_name:<15}  "
                  f"10th: {lo:7.2f} kWh | "
                  f"50th: {med:7.2f} kWh | "
                  f"90th: {hi:7.2f} kWh | "
                  f"Uncertainty: {u:.4f}")

    print_uncertainty_table(uncertainty)
    plot_forecast(pred_raw, y_raw, uncertainty)
    print("[inference] Done.")


if __name__ == "__main__":
    main()
