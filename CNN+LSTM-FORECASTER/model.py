"""
model.py
--------
Physics-constrained CNN-LSTM architecture for multi-output probabilistic
energy forecasting.

Architecture overview:
  Input  (batch, SEQ_LEN, n_features)
    |
    ├- CNN block (2 x Conv1d with kernel=3)
    |    Extracts short-range local patterns (e.g. morning ramp-up).
    |
    ├- LSTM block (2 stacked LSTM layers, hidden=128)
    |    Captures long-range temporal dependencies across the 48-h window.
    |
    └- Quantile head (Dense)
         Outputs 3 targets x 3 quantiles = 9 values per timestep.
         Repeated for each of the HORIZON forecast steps.

Note on soft physics clamping
  A "soft physics clamp" is applied in the forward pass to bias predictions
  toward feasible regions BEFORE the loss sees them.  The hard penalty in
  the loss function is a second safety net for any remaining violations.
"""

import torch
import torch.nn as nn
from config import (
    CNN_CHANNELS, CNN_KERNEL_SIZE, LSTM_HIDDEN, LSTM_LAYERS,
    DROPOUT, QUANTILES, HORIZON,
    BETZ_COEFF, AIR_DENSITY, ROTOR_AREA,
    IDX_SOLAR, IDX_WIND, IDX_LOAD,
    Q_LOW, Q_MID, Q_HIGH,
)


class CNNBlock(nn.Module):
    """
    Two Conv1d layers that operate on the time dimension.
    Conv1d expects input shape (batch, channels, seq_len).
    We transpose the input, apply convolutions, then transpose back.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        c1, c2 = CNN_CHANNELS

        # Layer 1
        self.conv1 = nn.Conv1d(in_channels, c1,
                               kernel_size=CNN_KERNEL_SIZE, padding="same")
        self.bn1   = nn.BatchNorm1d(c1)
        self.act1  = nn.GELU()
        self.drop1 = nn.Dropout(DROPOUT)

        # Layer 2
        self.conv2 = nn.Conv1d(c1, c2,
                               kernel_size=CNN_KERNEL_SIZE, padding="same")
        self.bn2   = nn.BatchNorm1d(c2)
        self.act2  = nn.GELU()
        self.drop2 = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        x = x.permute(0, 2, 1)          # -> (batch, features, seq_len)
        x = self.drop1(self.act1(self.bn1(self.conv1(x))))
        x = self.drop2(self.act2(self.bn2(self.conv2(x))))
        x = x.permute(0, 2, 1)          # -> (batch, seq_len, channels)
        return x


class LSTMBlock(nn.Module):
    """
    Two stacked LSTM layers with dropout between them.
    Returns the full output sequence (not just the final hidden state)
    so the quantile head can produce per-step forecasts for the horizon.
    """
    def __init__(self, in_size: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = in_size,
            hidden_size = LSTM_HIDDEN,
            num_layers  = LSTM_LAYERS,
            dropout     = DROPOUT,      # dropout between stacked layers
            batch_first = True,
        )
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor):
        # x: (batch, seq_len, in_size)
        out, _ = self.lstm(x)           # out: (batch, seq_len, hidden)
        return self.dropout(out)


class QuantileHead(nn.Module):
    """
    Maps LSTM hidden representation -> quantile predictions.
    
    We use only the LAST hidden state from the encoder to decode into the
    full HORIZON future steps.  This is a "direct multi-step" strategy:
    one forward pass produces all H timesteps simultaneously, avoiding the
    compounding errors of an autoregressive approach.

    Output shape: (batch, HORIZON, n_targets * n_quantiles)
    """
    def __init__(self):
        super().__init__()
        n_targets  = 3                           # solar, wind, load
        n_quantiles = len(QUANTILES)             # 3
        out_dim    = HORIZON * n_targets * n_quantiles

        self.fc = nn.Sequential(
            nn.Linear(LSTM_HIDDEN, 256),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, out_dim),
        )
        self.n_targets   = n_targets
        self.n_quantiles = n_quantiles

    def forward(self, h_last: torch.Tensor) -> torch.Tensor:
        # h_last: (batch, LSTM_HIDDEN)  — last timestep of LSTM output
        out = self.fc(h_last)                    # (batch, H * 3 * 3)
        out = out.view(h_last.size(0),
                       HORIZON,
                       self.n_targets,
                       self.n_quantiles)         # (batch, H, 3, 3)
        return out


class CNNLSTMForecaster(nn.Module):
    """
    Full CNN-LSTM forecasting model.

    Forward signature:
        x      : (batch, SEQ_LEN, n_features)   normalised input
        x_raw  : (batch, SEQ_LEN, n_features)   raw un-normalised input
                 (passed through only for the physics-constrained loss;
                  not used inside the model itself)

    Returns:
        preds  : (batch, HORIZON, 3, 3)
                 Axis -2 -> target  [IDX_SOLAR=0, IDX_WIND=1, IDX_LOAD=2]
                 Axis -1 -> quantile [Q_LOW=0, Q_MID=1, Q_HIGH=2]
    """

    def __init__(self, n_features: int):
        super().__init__()
        cnn_out   = CNN_CHANNELS[-1]             # 64

        self.cnn  = CNNBlock(n_features)
        self.lstm = LSTMBlock(cnn_out)
        self.head = QuantileHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z      = self.cnn(x)                       # (batch, seq, cnn_out)
        z      = self.lstm(z)                      # (batch, seq, hidden)
        h_last = z[:, -1, :]                       # (batch, hidden)
        preds  = self.head(h_last)                 # (batch, H, 3, Q)

        # -- Quantile sorting (fix for collapse bug) -----------------------
        # Softplus caused all 3 quantiles to collapse to the same constant.
        # Sorting enforces Q_LOW <= Q_MID <= Q_HIGH without restricting range,
        # which is essential for non-zero prediction intervals.
        preds, _ = torch.sort(preds, dim=-1)       # sort along quantile axis
        return preds


def _soft_clamp(preds: torch.Tensor) -> torch.Tensor:
    """
    DEPRECATED — replaced by quantile sorting in CNNLSTMForecaster.forward().
    Softplus caused model collapse: all 3 quantiles converged to the same
    constant value (~log(2)) making uncertainty = 0 everywhere.
    Kept here for reference only.
    """
    return torch.nn.functional.softplus(preds)
