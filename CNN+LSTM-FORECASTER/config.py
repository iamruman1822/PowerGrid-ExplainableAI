"""
config.py
---------
Central configuration for the CNN-LSTM physics-constrained energy forecaster.
Calibrated to:
  Location : Gwalior, Lakhnotikhurd, Madhya Pradesh, India
  Lat/Lon  : 26.193 N, 78.1487 E
  Standards: MNRE village microgrid | CEA Grid Code 2023 | BIS IS 14286

All hyperparameters, paths, and physics constants are defined here.
"""

# -- Data & Paths --------------------------------------------------------------
# After running fetch_data.py this points to real Open-Meteo weather
DATA_CSV           = "data/gwalior_microgrid.csv"
MODEL_SAVE_PATH    = "outputs/cnn_lstm_forecaster.pt"
LOSS_PLOT_PATH     = "outputs/loss_curve.png"
FORECAST_PLOT_PATH = "outputs/sample_forecast.png"

# -- Location Metadata ---------------------------------------------------------
LOCATION_NAME      = "Gwalior, Lakhnotikhurd, Madhya Pradesh"
LATITUDE           = 26.193     # degrees N
LONGITUDE          = 78.1487    # degrees E
TIMEZONE           = "Asia/Kolkata"   # IST = UTC+5:30

# -- Sequence Parameters -------------------------------------------------------
SEQ_LEN  = 48   # lookback window (hours)
HORIZON  = 24   # forecast horizon (hours)

# -- Feature Columns -----------------------------------------------------------
FEATURE_COLS = [
    "GHI",         # W/m2   – Global Horizontal Irradiance  (Open-Meteo real)
    "wind_speed",  # m/s    – Wind speed at 10 m hub height (Open-Meteo real)
    "temperature", #  degC     – Ambient temperature           (Open-Meteo real)
    "humidity",    # %      – Relative humidity             (Open-Meteo real)
    "hour",        # 0–23   – IST hour  (cyclic -> 2 cols in dataset.py)
    "day_of_year", # 1–365  (cyclic -> 2 cols in dataset.py)
    # + 48 load-lag features appended in dataset.py
]
TARGET_COLS = ["solar_power", "wind_power", "load_demand"]  # kWh

# -- Training Hyperparameters --------------------------------------------------
BATCH_SIZE    = 32
EPOCHS        = 150      # more room to converge
PATIENCE      = 15       # wider patience after physics kicks in
WARMUP_EPOCHS = 20       # first N epochs: pinball only, no physics penalty
LEARNING_RATE = 5e-4     # lower LR for stable convergence
TRAIN_FRAC    = 0.70
VAL_FRAC      = 0.15
# TEST_FRAC   = 0.15  (remainder)

# -- Model Architecture --------------------------------------------------------
CNN_CHANNELS    = [32, 64]
CNN_KERNEL_SIZE = 3
LSTM_HIDDEN     = 128
LSTM_LAYERS     = 2
DROPOUT         = 0.2
QUANTILES       = [0.10, 0.50, 0.90]

# -- Physics Constants — MNRE / IEC 61400 / BIS -------------------------------

# - Solar (BIS IS 14286, MNRE village DRE scheme) -----------------------------
# 25 kWp community array: 139 panels x 180 Wp at 18% efficiency
# Area = 25000 W / (1000 W/m2 * 0.18) = 138.9 m2
SOLAR_PANEL_AREA      = 138.9       # m2  <- 25 kWp array
SOLAR_PANEL_EFF       = 0.18        # 18% (BIS IS 14286 Grade A poly/mono)
SOLAR_TEMP_COEFF      = -0.004      # -0.4%/ degC (standard NOCT power derating)
SOLAR_NOCT_TEMP       = 25.0        #  degC reference temperature
# Clear-sky GHI ceiling used in physics loss
# ceiling = EFF * AREA * GHI / 1000  (kWh)

# - Wind (MNRE small wind turbine, IEC 61400-2, e.g. Suzlon S-20 / W-2000) --
# 10 kW turbine, rotor diameter = 7 m -> Area = π*(3.5)2 = 38.48 m2
BETZ_COEFF      = 0.593             # Betz limit
AIR_DENSITY     = 1.225             # kg/m3  (sea level, adjust for elevation)
ROTOR_AREA      = 38.48             # m2  (7 m diameter turbine)
ROTOR_DIAMETER  = 7.0               # m
WIND_CUT_IN     = 3.0               # m/s  cut-in speed
WIND_RATED      = 11.0              # m/s  rated speed (full power)
WIND_CUT_OUT    = 25.0              # m/s  cut-out (turbine shuts down)

# - Load (MNRE Tier-2 village electrification, Lakhnotikhurd) -----------------
# ~200 HH x 500W average + agri pumps + street lights
LOAD_MIN_KW     = 20.0              # kW  absolute floor (non-negotiable minimum)

# - General constraint penalty ------------------------------------------------
PHYSICS_PENALTY = 0.1               # multiplier on violated constraint
                                    # NOTE: 10.0 caused model collapse — violations
                                    # overwhelmed pinball loss by 5 orders of magnitude

# -- Quantile indices (do not change) -----------------------------------------
Q_LOW  = 0   # 10th percentile
Q_MID  = 1   # 50th percentile (median forecast)
Q_HIGH = 2   # 90th percentile

# -- Output target indices -----------------------------------------------------
IDX_SOLAR = 0
IDX_WIND  = 1
IDX_LOAD  = 2

# -- Random seed ---------------------------------------------------------------
SEED = 42
