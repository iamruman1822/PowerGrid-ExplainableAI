"""
utils.py
--------
Plotting helpers and miscellaneous utilities.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")    # non-interactive backend (safe for servers / Windows)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from config import (
    LOSS_PLOT_PATH, FORECAST_PLOT_PATH,
    HORIZON, QUANTILES, Q_LOW, Q_MID, Q_HIGH,
    IDX_SOLAR, IDX_WIND, IDX_LOAD,
)

TARGET_NAMES = ["Solar Power (kWh)", "Wind Power (kWh)", "Load Demand (kWh)"]
COLORS       = ["#F6AE2D", "#2D9CDB", "#6C63FF"]   # gold, blue, purple


def ensure_output_dir():
    os.makedirs("outputs", exist_ok=True)


# -----------------------------------------------------------------------------
# Loss curve
# -----------------------------------------------------------------------------

def plot_loss_curve(train_losses: list[float],
                    val_losses  : list[float]) -> None:
    """
    Saves a two-line (train / val) loss curve to LOSS_PLOT_PATH.
    """
    ensure_output_dir()
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d2e")
    ax.spines[:].set_color("#2a2d3e")
    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.title.set_color("white")

    ax.plot(epochs, train_losses, color="#F6AE2D", linewidth=2, label="Train loss")
    ax.plot(epochs, val_losses,   color="#2D9CDB", linewidth=2,
            linestyle="--", label="Val loss")
    ax.fill_between(epochs, train_losses,
                    alpha=0.15, color="#F6AE2D")
    ax.fill_between(epochs, val_losses,
                    alpha=0.15, color="#2D9CDB")

    ax.set_xlabel("Epoch",         fontsize=12)
    ax.set_ylabel("Loss",          fontsize=12)
    ax.set_title("Training & Validation Loss (Pinball + Physics Penalty)",
                 fontsize=13, pad=12)
    ax.legend(facecolor="#1a1d2e", labelcolor="white", framealpha=0.8)
    ax.grid(True, linestyle="--", alpha=0.3, color="white")

    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[utils] Loss curve saved -> {LOSS_PLOT_PATH}")


# -----------------------------------------------------------------------------
# Forecast plot (3-panel)
# -----------------------------------------------------------------------------

def plot_forecast(pred_raw   : np.ndarray,
                  y_raw      : np.ndarray,
                  uncertainty: np.ndarray) -> None:
    """
    Plots a 3x2 grid:  left column -> forecast with confidence band
                        right column -> uncertainty score per timestep.

    pred_raw   : (HORIZON, 3, 3)   raw kWh  [target, quantile]
    y_raw      : (HORIZON, 3)      raw kWh  ground truth
    uncertainty: (HORIZON, 3)      [0, 1]
    """
    ensure_output_dir()
    hours = np.arange(1, HORIZON + 1)

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#0f1117")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    for i, (name, color) in enumerate(zip(TARGET_NAMES, COLORS)):
        ax_fc   = fig.add_subplot(gs[i, 0])
        ax_unc  = fig.add_subplot(gs[i, 1])

        for ax in [ax_fc, ax_unc]:
            ax.set_facecolor("#1a1d2e")
            ax.spines[:].set_color("#2a2d3e")
            ax.tick_params(colors="white")
            ax.yaxis.label.set_color("white")
            ax.xaxis.label.set_color("white")
            ax.title .set_color("white")
            ax.grid(True, linestyle="--", alpha=0.25, color="white")

        low  = pred_raw[:, i, Q_LOW]
        mid  = pred_raw[:, i, Q_MID]
        high = pred_raw[:, i, Q_HIGH]
        gt   = y_raw   [:, i]
        unc  = uncertainty[:, i]

        # Left – forecast
        ax_fc.fill_between(hours, low, high,
                           alpha=0.25, color=color, label="90% band")
        ax_fc.plot(hours, mid, color=color, linewidth=2.5, label="Predicted (50th)")
        ax_fc.plot(hours, gt,  color="white", linewidth=1.5,
                   linestyle="--", label="Actual")
        ax_fc.set_title(f"{name} – 24 h Forecast", fontsize=11)
        ax_fc.set_xlabel("Hour ahead", fontsize=9)
        ax_fc.set_ylabel("kWh", fontsize=9)
        ax_fc.legend(facecolor="#1a1d2e", labelcolor="white",
                     fontsize=8, framealpha=0.8)

        # Right – uncertainty
        ax_unc.bar(hours, unc, color=color, alpha=0.7, edgecolor="none")
        ax_unc.set_ylim(0, 1)
        ax_unc.set_title(f"{name} – Uncertainty Score", fontsize=11)
        ax_unc.set_xlabel("Hour ahead", fontsize=9)
        ax_unc.set_ylabel("Uncertainty [0–1]", fontsize=9)

    fig.suptitle("24-Hour Energy Forecast — Physics-Constrained CNN-LSTM",
                 color="white", fontsize=14, y=0.99)

    plt.savefig(FORECAST_PLOT_PATH, dpi=150,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[utils] Forecast plot saved -> {FORECAST_PLOT_PATH}")


# -----------------------------------------------------------------------------
# Console reporting helpers
# -----------------------------------------------------------------------------

def print_uncertainty_table(uncertainty: np.ndarray) -> None:
    """
    Prints a formatted per-timestep uncertainty table to stdout.
    uncertainty : (HORIZON, 3)
    """
    headers = ["Hour", "Solar Uncert.", "Wind Uncert.", "Load Uncert."]
    widths  = [6, 14, 13, 13]
    sep     = "-" * (sum(widths) + 3 * len(widths))
    header  = " ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(f"\n{sep}")
    print(f"  24-HOUR UNCERTAINTY SCORES (0=certain, 1=maximum)")
    print(sep)
    print(f"  {header}")
    print(sep)
    for h in range(HORIZON):
        row = f"  {h+1:<6}"
        for t in range(3):
            row += f"{uncertainty[h, t]:.4f}".ljust(widths[t+1])
        print(row)
    print(sep + "\n")


def print_epoch_summary(epoch     : int,
                        train_loss: float,
                        val_loss  : float,
                        n_violations: int) -> None:
    bar_len = 30
    ratio   = min(val_loss / (train_loss + 1e-9), 2.0)
    filled  = int(bar_len * min(train_loss / 5.0, 1.0))
    bar     = "█" * filled + "░" * (bar_len - filled)

    print(f"Epoch {epoch:>3} | Train: {train_loss:.5f} | "
          f"Val: {val_loss:.5f} | [{'█'*filled}{'░'*(bar_len-filled)}] | "
          f"Violations: {n_violations}")
