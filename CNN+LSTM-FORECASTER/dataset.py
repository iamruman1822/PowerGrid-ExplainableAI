"""
dataset.py
----------
Handles all data loading, feature engineering, normalisation,
and PyTorch Dataset / DataLoader construction.

Key steps:
  1. Load CSV, parse timestamps, derive cyclic time features.
  2. Append 48 lag features for load demand (as specified).
  3. Normalise all features and targets with StandardScaler.
  4. Slice into (input_seq, target_seq) pairs using a sliding window.
  5. Split chronologically into train / val / test.
  6. Return DataLoaders and the fitted scalers (needed for inverse-transform).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from config import (
    DATA_CSV, SEQ_LEN, HORIZON,
    TRAIN_FRAC, VAL_FRAC, BATCH_SIZE, SEED
)


# -----------------------------------------------------------------------------
# 1.  Feature Engineering
# -----------------------------------------------------------------------------

def _add_cyclic(df: pd.DataFrame) -> pd.DataFrame:
    """Encode hour-of-day and day-of-year as sine/cosine pairs so the model
    sees that hour 23 and hour 0 are neighbours (cyclical continuity)."""
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"]  = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"]  = np.cos(2 * np.pi * df["day_of_year"] / 365)
    return df


def _add_lag_features(df: pd.DataFrame, n_lags: int = 48) -> pd.DataFrame:
    """Create lagged copies of load_demand (t-1 … t-48)."""
    df = df.copy()
    for lag in range(1, n_lags + 1):
        df[f"load_lag_{lag}"] = df["load_demand"].shift(lag)
    # Drop rows that contain NaN from lag creation
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_dataframe() -> pd.DataFrame:
    """Load and engineer features from CSV."""
    df = pd.read_csv(DATA_CSV, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Derive time fields
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_year"] = df["timestamp"].dt.day_of_year

    # Cyclic encoding of time
    df = _add_cyclic(df)

    # Lag features for load
    df = _add_lag_features(df, n_lags=48)

    return df


# -----------------------------------------------------------------------------
# 2.  Feature / Target column lists (built dynamically after lag creation)
# -----------------------------------------------------------------------------

STATIC_FEATURES = [
    "GHI", "wind_speed", "temperature", "humidity",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]
LAG_FEATURES = [f"load_lag_{i}" for i in range(1, 49)]
ALL_FEATURES = STATIC_FEATURES + LAG_FEATURES   # 8 + 48 = 56 features

TARGET_COLS  = ["solar_power", "wind_power", "load_demand"]

# Raw column indices needed by the physics-constrained loss.
# The loss receives the un-normalised input x_raw and extracts these columns.
GHI_RAW_IDX        = ALL_FEATURES.index("GHI")          # for solar ceiling
WIND_SPEED_RAW_IDX = ALL_FEATURES.index("wind_speed")   # for Betz / IEC 61400-2
TEMP_RAW_IDX       = ALL_FEATURES.index("temperature")  # for NOCT thermal derating


# -----------------------------------------------------------------------------
# 3.  Sliding-window Dataset
# -----------------------------------------------------------------------------

class EnergyDataset(Dataset):
    """
    Each sample:
      x  – (SEQ_LEN, n_features)   normalised input window
      y  – (HORIZON,  3)            normalised targets [solar, wind, load]
      x_raw – (SEQ_LEN, n_features) un-normalised input  (for physics loss)
      y_raw – (HORIZON,  3)         un-normalised targets (for physics loss)
    """

    def __init__(self,
                 X: np.ndarray,       # shape (T, n_features) – normalised
                 Y: np.ndarray,       # shape (T, 3)           – normalised
                 X_raw: np.ndarray,   # shape (T, n_features) – raw
                 Y_raw: np.ndarray):  # shape (T, 3)           – raw
        self.X     = torch.tensor(X,     dtype=torch.float32)
        self.Y     = torch.tensor(Y,     dtype=torch.float32)
        self.X_raw = torch.tensor(X_raw, dtype=torch.float32)
        self.Y_raw = torch.tensor(Y_raw, dtype=torch.float32)

        # Number of valid windows
        self.n_windows = len(X) - SEQ_LEN - HORIZON + 1

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        x_win     = self.X    [idx : idx + SEQ_LEN]             # (SEQ_LEN, F)
        y_win     = self.Y    [idx + SEQ_LEN : idx + SEQ_LEN + HORIZON]  # (H, 3)
        x_raw_win = self.X_raw[idx : idx + SEQ_LEN]
        y_raw_win = self.Y_raw[idx + SEQ_LEN : idx + SEQ_LEN + HORIZON]
        return x_win, y_win, x_raw_win, y_raw_win


# -----------------------------------------------------------------------------
# 4.  Build DataLoaders
# -----------------------------------------------------------------------------

def get_dataloaders():
    """
    Returns:
        train_loader, val_loader, test_loader  – PyTorch DataLoaders
        feat_scaler                            – fitted StandardScaler for features
        tgt_scaler                             – fitted StandardScaler for targets
        ghi_raw_idx, wind_raw_idx              – column indices (int)
    """
    df = load_dataframe()

    # -- Raw arrays ------------------------------------------------------------
    X_raw = df[ALL_FEATURES].values.astype(np.float32)   # (T, 56)
    Y_raw = df[TARGET_COLS ].values.astype(np.float32)   # (T, 3)

    # -- Chronological split indices -------------------------------------------
    T       = len(df)
    i_train = int(T * TRAIN_FRAC)
    i_val   = int(T * (TRAIN_FRAC + VAL_FRAC))

    # -- Fit scalers on training slice only to prevent leakage ----------------
    feat_scaler = StandardScaler()
    tgt_scaler  = StandardScaler()

    feat_scaler.fit(X_raw[:i_train])
    tgt_scaler .fit(Y_raw[:i_train])

    X_norm = feat_scaler.transform(X_raw).astype(np.float32)
    Y_norm = tgt_scaler .transform(Y_raw).astype(np.float32)

    # -- Create datasets -------------------------------------------------------
    def make_ds(start, end):
        return EnergyDataset(
            X_norm [start:end],
            Y_norm [start:end],
            X_raw  [start:end],
            Y_raw  [start:end],
        )

    train_ds = make_ds(0,       i_train)
    val_ds   = make_ds(i_train, i_val)
    test_ds  = make_ds(i_val,   T)

    # Generator for reproducible shuffle
    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  generator=g, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, drop_last=False)

    print(f"[dataset] Train: {len(train_ds):,} | Val: {len(val_ds):,} | "
          f"Test: {len(test_ds):,} windows")
    print(f"[dataset] Feature dim: {X_norm.shape[1]} | Targets: {TARGET_COLS}")

    return (train_loader, val_loader, test_loader,
            feat_scaler, tgt_scaler,
            GHI_RAW_IDX, WIND_SPEED_RAW_IDX)
