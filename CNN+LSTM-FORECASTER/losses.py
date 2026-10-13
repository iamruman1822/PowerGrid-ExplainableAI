"""
losses.py
---------
Physics-constrained quantile loss (Pinball + hard penalty).
Calibrated to Gwalior, MP microgrid — MNRE / IEC 61400-2 / BIS IS 14286.

Quantile (Pinball) Loss:
  For each quantile q ∈ {0.10, 0.50, 0.90} and each target variable,
  the pinball loss is:
    L(q, y, ŷ) = q*(y - ŷ)        if y >= ŷ
               = (q-1)*(y - ŷ)    if y <  ŷ

Physics Constraints (each violation adds 10x the violated amount to loss):
  1. Solar ceiling (NOCT-corrected)
     ceiling = eta(T) x A_panel x GHI / 1000  (kWh)
     eta(T) = eta_ref x [1 + alphax(T_cell - T_ref)]
     T_cell = T_amb + (NOCT-20)/800 x GHI   (IEC 61215 NOCT model)

  2. Wind ceiling (IEC 61400-2 piecewise curve)
     v < cut_in              -> P = 0
     cut_in <= v < rated      -> P = Betz x 0.5 x rho x A x v3 / 1000
     rated  <= v < cut_out   -> P = P_rated (constant)
     v >= cut_out             -> P = 0  (turbine shuts down)

  3. Load floor: load >= LOAD_MIN_KW (Gwalior base load is never zero)

NOTE:
  The loss receives *normalised* predictions and *normalised* targets.
  Physics ceilings are computed in raw space using the tgt_scaler.
"""

import torch
import torch.nn as nn
import numpy as np
from config import (
    QUANTILES, PHYSICS_PENALTY,
    BETZ_COEFF, AIR_DENSITY, ROTOR_AREA,
    WIND_CUT_IN, WIND_RATED, WIND_CUT_OUT,
    SOLAR_PANEL_AREA, SOLAR_PANEL_EFF, SOLAR_TEMP_COEFF, SOLAR_NOCT_TEMP,
    LOAD_MIN_KW,
    IDX_SOLAR, IDX_WIND, IDX_LOAD,
    Q_LOW, Q_MID, Q_HIGH,
)

# NOCT (Nominal Operating Cell Temperature) per IEC 61215
NOCT = 45.0   #  degC


class PhysicsConstrainedQuantileLoss(nn.Module):
    """
    Combines pinball loss and physics-violation penalties.

    Args:
        tgt_scaler   : fitted sklearn StandardScaler for targets.
                       Used to convert predictions back to raw kWh for
                       physics ceiling comparison.
        violation_counter : list[int] — accumulates violation counts per
                       epoch; pass an empty list and check after each epoch.
    """

    def __init__(self, tgt_scaler, violation_counter: list):
        super().__init__()
        self.quantiles          = QUANTILES                 # [0.10, 0.50, 0.90]
        self.tgt_scaler         = tgt_scaler
        self.violation_counter  = violation_counter         # mutable reference

        # Cache scaler stats as torch tensors for GPU-safe inverse transform
        mean  = torch.tensor(tgt_scaler.mean_ , dtype=torch.float32)
        scale = torch.tensor(tgt_scaler.scale_, dtype=torch.float32)
        self.register_buffer = lambda name, t: None         # noop (not nn.Module)
        self._tgt_mean  = mean    # shape (3,)
        self._tgt_scale = scale   # shape (3,)

    # -- helpers --------------------------------------------------------------

    def _to_raw_targets(self, pred_norm: torch.Tensor) -> torch.Tensor:
        """
        Inverse-normalise predictions.
        pred_norm : (batch, H, 3, Q) normalised
        returns   : (batch, H, 3, Q) raw kWh
        """
        m = self._tgt_mean .to(pred_norm.device)   # (3,)
        s = self._tgt_scale.to(pred_norm.device)   # (3,)
        # broadcast along H and Q axes
        return pred_norm * s[None, None, :, None] + m[None, None, :, None]

    # -- pinball loss ----------------------------------------------------------

    def _pinball(self,
                 pred_norm: torch.Tensor,
                 y_norm   : torch.Tensor) -> torch.Tensor:
        """
        pred_norm : (batch, H, 3, Q)
        y_norm    : (batch, H, 3)    — ground truth

        Returns scalar mean pinball loss.
        """
        # Expand y to match quantile dim: (batch, H, 3, 1)
        y = y_norm.unsqueeze(-1)
        # residuals: positive where y > ŷ, negative where y < ŷ
        residuals = y - pred_norm                   # (batch, H, 3, Q)

        q = torch.tensor(self.quantiles,
                         dtype=torch.float32,
                         device=pred_norm.device)   # (Q,)
        # Pinball: max(q*r, (q-1)*r)
        loss = torch.max(q * residuals, (q - 1) * residuals)
        return loss.mean()

    # -- physics penalties -----------------------------------------------------

    def _solar_penalty(self,
                       pred_norm : torch.Tensor,
                       x_raw     : torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Solar ceiling with NOCT temperature derating (IEC 61215):
          T_cell   = T_amb + (NOCT-20)/800 * GHI
          eta(T)     = eta_ref * [1 + alpha*(T_cell - T_ref)]
          ceiling  = max(eta(T), 0.05) * A * GHI / 1000  (kWh)
        GHI and temperature taken from the LAST input timestep.
        """
        from dataset import GHI_RAW_IDX, TEMP_RAW_IDX

        pred_raw  = self._to_raw_targets(pred_norm)           # (B, H, 3, Q)
        ghi_last  = x_raw[:, -1, GHI_RAW_IDX]                # (B,)
        temp_last = x_raw[:, -1, TEMP_RAW_IDX]               # (B,)

        # NOCT cell temperature
        t_cell  = temp_last + (NOCT - 20.0) / 800.0 * ghi_last
        eff_adj = SOLAR_PANEL_EFF * (
                    1.0 + SOLAR_TEMP_COEFF * (t_cell - SOLAR_NOCT_TEMP))
        eff_adj = torch.clamp(eff_adj, min=0.05)              # floor at 5%

        ceiling    = eff_adj * SOLAR_PANEL_AREA * ghi_last / 1000.0
        ceiling    = ceiling[:, None, None]                   # (B, 1, 1)

        solar_pred = pred_raw[:, :, IDX_SOLAR, :]             # (B, H, Q)
        violations = torch.clamp(solar_pred - ceiling, min=0)
        n_viol     = int((violations > 0).sum().item())
        penalty    = PHYSICS_PENALTY * violations.mean()
        return penalty, n_viol

    def _betz_penalty(self,
                      pred_norm : torch.Tensor,
                      x_raw     : torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        IEC 61400-2 piecewise wind turbine power curve ceiling:
          v < cut_in              -> ceiling = 0
          cut_in <= v < rated      -> ceiling = Betz x 0.5 x rho x A x v3 / 1000
          rated  <= v < cut_out   -> ceiling = P_rated
          v >= cut_out             -> ceiling = 0  (turbine shutdown)
        Wind speed from the last input timestep.
        """
        from dataset import WIND_SPEED_RAW_IDX

        pred_raw = self._to_raw_targets(pred_norm)             # (B, H, 3, Q)
        v        = x_raw[:, -1, WIND_SPEED_RAW_IDX]           # (B,)

        p_aero   = (BETZ_COEFF * 0.5 * AIR_DENSITY * ROTOR_AREA
                    * v.pow(3) / 1000.0)
        p_rated  = torch.full_like(v,
                       BETZ_COEFF * 0.5 * AIR_DENSITY * ROTOR_AREA
                       * WIND_RATED ** 3 / 1000.0)

        # Piecewise ceiling per sample
        ceiling = torch.where(v < WIND_CUT_IN,  torch.zeros_like(v),
                  torch.where(v < WIND_RATED,   p_aero,
                  torch.where(v < WIND_CUT_OUT, p_rated,
                                                torch.zeros_like(v))))
        ceiling = ceiling[:, None, None]                       # (B, 1, 1)

        wind_pred  = pred_raw[:, :, IDX_WIND, :]              # (B, H, Q)
        violations = torch.clamp(wind_pred - ceiling, min=0)
        n_viol     = int((violations > 0).sum().item())
        penalty    = PHYSICS_PENALTY * violations.mean()
        return penalty, n_viol

    def _load_floor_penalty(self,
                            pred_norm: torch.Tensor
                            ) -> tuple[torch.Tensor, int]:
        """
        Gwalior village microgrid always has a minimum base load.
        Load predictions must be >= LOAD_MIN_KW at every timestep.
        Solar & Wind must be >= 0 (non-negativity).
        """
        pred_raw = self._to_raw_targets(pred_norm)             # (B, H, 3, Q)

        # Non-negativity for solar & wind
        solar_wind  = pred_raw[:, :, [IDX_SOLAR, IDX_WIND], :]
        neg_viol    = torch.clamp(-solar_wind, min=0)

        # Load floor
        load_pred   = pred_raw[:, :, IDX_LOAD, :]             # (B, H, Q)
        floor_viol  = torch.clamp(LOAD_MIN_KW - load_pred, min=0)

        all_viol   = torch.cat([neg_viol.reshape(-1),
                                floor_viol.reshape(-1)])
        n_viol     = int((all_viol > 0).sum().item())
        penalty    = PHYSICS_PENALTY * all_viol.mean()
        return penalty, n_viol

    # -- main forward ----------------------------------------------------------

    def forward(self,
                pred_norm : torch.Tensor,
                y_norm    : torch.Tensor,
                x_raw     : torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_norm : (batch, H, 3, Q)  – model output (normalised)
            y_norm    : (batch, H, 3)     – ground-truth targets (normalised)
            x_raw     : (batch, SEQ_LEN, n_features) – raw (un-normalised) input
        
        Returns:
            total_loss : scalar tensor
        """
        # 1. Base quantile (pinball) loss
        pinball_loss = self._pinball(pred_norm, y_norm)

        # 2. Physics violation penalties
        solar_pen,  n_solar = self._solar_penalty    (pred_norm, x_raw)
        betz_pen,   n_betz  = self._betz_penalty     (pred_norm, x_raw)
        floor_pen,  n_floor = self._load_floor_penalty(pred_norm)

        # Accumulate violation counts for reporting
        total_violations = n_solar + n_betz + n_floor
        self.violation_counter.append(total_violations)

        total_loss = pinball_loss + solar_pen + betz_pen + floor_pen
        return total_loss


# -- Uncertainty score ---------------------------------------------------------

def compute_uncertainty(pred_raw: torch.Tensor) -> torch.Tensor:
    """
    Uncertainty = (upper - lower) / (mean + ε), clipped to [0, 1].

    pred_raw : (H, 3, 3) raw kWh  [target, quantile]
    Returns  : (H, 3)             uncertainty score per timestep per target
    """
    eps   = 1e-6
    lower = pred_raw[:, :, Q_LOW]    # (H, 3)
    mean  = pred_raw[:, :, Q_MID]    # (H, 3)
    upper = pred_raw[:, :, Q_HIGH]   # (H, 3)

    uncertainty = (upper - lower) / (mean.abs() + eps)
    return uncertainty.clamp(0.0, 1.0)
