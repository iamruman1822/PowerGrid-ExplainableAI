# CNN-LSTM Physics-Constrained Probabilistic Energy Forecaster

Predict the next **24 hours** of solar power, wind power, and load demand —
each with 10th / 50th / 90th percentile confidence bands and an uncertainty score —
using a physics-constrained CNN-LSTM trained with quantile (pinball) regression loss.

---

## Project Structure

```
CNN+LSTM-FORECASTER/
├── config.py           ← All hyperparameters & physics constants
├── generate_data.py    ← Synthetic 2-year hourly CSV generator
├── dataset.py          ← Feature engineering, normalisation, DataLoaders
├── model.py            ← CNN-LSTM architecture
├── losses.py           ← Pinball loss + physics-penalty loss
├── utils.py            ← Plotting & console reporting
├── train.py            ← Full training pipeline
├── inference.py        ← Standalone 24-h forecast (post-training)
├── requirements.txt
├── data/
│   └── synthetic_energy_data.csv  (auto-generated)
└── outputs/
    ├── cnn_lstm_forecaster.pt     (saved model)
    ├── loss_curve.png
    └── sample_forecast.png
```

---

## Quick Start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** Uncomment the CUDA line in `requirements.txt` and install the
> CUDA-enabled wheel instead.

### 2 — Run training

```bash
python train.py
```

This will:
- Auto-generate `data/synthetic_energy_data.csv` (if not present)
- Train for up to 100 epochs with early stopping (patience=10)
- Print physics violation count after each epoch
- Save the best model to `outputs/cnn_lstm_forecaster.pt`
- Save `outputs/loss_curve.png` and `outputs/sample_forecast.png`

### 3 — Run inference on new data

```bash
python inference.py
# or with your own CSV:
python inference.py --csv path/to/your_data.csv
```

---

## Using Your Own Dataset

Your CSV must have **at minimum** these columns (hourly, UTC timestamps):

| Column        | Units | Description                    |
|---------------|-------|--------------------------------|
| `timestamp`   | —     | `YYYY-MM-DD HH:MM:SS`          |
| `GHI`         | W/m²  | Global Horizontal Irradiance   |
| `wind_speed`  | m/s   | Wind speed at 10 m hub height  |
| `temperature` | °C    | Ambient temperature            |
| `humidity`    | %     | Relative humidity              |
| `load_demand` | kWh   | Actual load demand             |
| `solar_power` | kWh   | Actual solar generation        |
| `wind_power`  | kWh   | Actual wind generation         |

Place it at `data/synthetic_energy_data.csv` or update `DATA_CSV` in `config.py`.

---

## Key Configuration (`config.py`)

| Parameter        | Default | Description                          |
|------------------|---------|--------------------------------------|
| `SEQ_LEN`        | 48      | Lookback window (hours)              |
| `HORIZON`        | 24      | Forecast horizon (hours)             |
| `BATCH_SIZE`     | 32      | Mini-batch size                      |
| `EPOCHS`         | 100     | Maximum training epochs              |
| `PATIENCE`       | 10      | Early-stopping patience              |
| `LEARNING_RATE`  | 0.001   | Adam initial LR                      |
| `LSTM_HIDDEN`    | 128     | LSTM hidden units                    |
| `DROPOUT`        | 0.2     | Dropout probability                  |
| `ROTOR_AREA`     | 50.0    | Wind turbine swept area (m²) — **set this to your turbine** |
| `PHYSICS_PENALTY`| 10.0    | Multiplier for constraint violations |

---

## Physics Constraints

Three constraints are embedded directly into the loss function:

1. **Solar ceiling** — Prediction ≤ `η × A_panel × GHI / 1000` kWh  
   *(panel efficiency 18 %, area 200 m² — adjust in `generate_data.py`)*

2. **Betz limit** — Wind power ≤ `0.593 × 0.5 × 1.225 × A_rotor × v³ / 1000` kWh  
   *(set `ROTOR_AREA` in `config.py` to match your turbine)*

3. **Non-negativity** — All outputs ≥ 0  
   *(enforced softly via Softplus output activation + hard penalty)*

Any violation adds **10×** the violated amount to the batch loss.

---

## Output Interpretation

```
Hour +01:
  solar_power     10th:  12.30 kWh │ 50th:  18.45 kWh │ 90th:  24.60 kWh │ Uncertainty: 0.6652
  wind_power      10th:   4.10 kWh │ 50th:   6.89 kWh │ 90th:   9.68 kWh │ Uncertainty: 0.8088
  load_demand     10th: 290.00 kWh │ 50th: 305.50 kWh │ 90th: 321.00 kWh │ Uncertainty: 0.1013
```

- **10th / 90th** → confidence band (how uncertain the model is)  
- **50th** → median prediction (use this as your point forecast)  
- **Uncertainty** → `(90th − 10th) / 50th`, clipped to [0, 1]; closer to 0 = confident

---

## Data Sources (if you want real data instead of synthetic)

| Source | What you get | Link |
|--------|-------------|------|
| **PVGIS** (EU) | Hourly GHI, temp by lat/lon | https://re.jrc.ec.europa.eu/pvg_tools/en/ |
| **ERA5** (global) | Wind, temp, humidity reanalysis | https://cds.climate.copernicus.eu |
| **Open-Meteo** | Free weather API, no key | https://open-meteo.com |
| **NREL NSRDB** | US solar resource data | https://nsrdb.nrel.gov |
| **ENTSO-E** | European electricity demand | https://transparency.entsoe.eu |

---

## What I Need From You (see companion message)

- Your turbine's **rotor swept area** (m²) or diameter (m)
- Your **solar panel** total area (m²) and efficiency rating
- Your **geographic location** (lat/lon) to calibrate GHI curves
- Whether you have a **real dataset** or want to use the synthetic one
