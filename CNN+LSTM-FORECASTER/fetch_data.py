"""
fetch_data.py
-------------
Downloads REAL data for Gwalior, Madhya Pradesh from:

  1. Open-Meteo Historical API (free, no key)
     - GHI (shortwave radiation), wind speed, temperature, humidity
     - 2 years: 2023-01-01 to 2024-12-31, hourly, IST

  2. Physics-derived power columns (since no public per-village metered data):
     - solar_power  : IEC 61215 NOCT model   (25 kWp MNRE array)
     - wind_power   : IEC 61400-2 power curve (10 kW, 7m rotor)
     - load_demand  : IMD-calibrated Indian village profile (MNRE Tier-2)
       The load profile uses REAL temperature/humidity from Open-Meteo
       to scale cooling/heating loads accurately.

Output:
    data/gwalior_microgrid.csv    <- real weather + physics power

Run:
    python fetch_data.py
"""

import urllib.request
import urllib.error
import json
import os
import sys
import time
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Import project config (physics constants already defined there)
# ─────────────────────────────────────────────────────────────────────────────
from config import (
    DATA_CSV, SEED,
    SOLAR_PANEL_AREA, SOLAR_PANEL_EFF, SOLAR_TEMP_COEFF, SOLAR_NOCT_TEMP,
    BETZ_COEFF, AIR_DENSITY, ROTOR_AREA,
    WIND_CUT_IN, WIND_RATED, WIND_CUT_OUT,
    LOAD_MIN_KW,
)

np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Open-Meteo API settings
# ─────────────────────────────────────────────────────────────────────────────
LAT = 26.193
LON = 78.1487
TZ  = "Asia/Kolkata"

OPEN_METEO_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude={lat}&longitude={lon}"
    "&start_date={start}&end_date={end}"
    "&hourly=shortwave_radiation,wind_speed_10m,"
    "temperature_2m,relative_humidity_2m"
    "&timezone={tz}"
    "&wind_speed_unit=ms"
)

NOCT = 45.0   # Nominal Operating Cell Temperature (IEC 61215)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: fetch one year
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_year(year: int, retries: int = 3) -> dict:
    url = OPEN_METEO_URL.format(
        lat=LAT, lon=LON,
        start=f"{year}-01-01", end=f"{year}-12-31",
        tz=TZ.replace("/", "%2F"),
    )
    print(f"  Downloading {year} from Open-Meteo ...")
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.URLError as e:
            print(f"  [attempt {attempt}/{retries}] Error: {e}")
            if attempt < retries:
                time.sleep(3)
    print("  FAILED to download. Check your internet connection.")
    sys.exit(1)


def _to_df(d: dict) -> pd.DataFrame:
    h = d["hourly"]
    return pd.DataFrame({
        "timestamp":   h["time"],
        "GHI":         h["shortwave_radiation"],
        "wind_speed":  h["wind_speed_10m"],
        "temperature": h["temperature_2m"],
        "humidity":    h["relative_humidity_2m"],
    })


# ─────────────────────────────────────────────────────────────────────────────
# Solar power — IEC 61215 NOCT model
# ─────────────────────────────────────────────────────────────────────────────
def _solar_power(ghi: np.ndarray, temp: np.ndarray) -> np.ndarray:
    """
    T_cell = T_amb + (NOCT - 20) / 800 * GHI
    eta(T) = eta_ref * [1 + alpha * (T_cell - T_ref)]
    P      = max(eta(T), 5%) * A * GHI / 1000   (kWh/h)
    """
    t_cell  = temp + (NOCT - 20.0) / 800.0 * ghi
    eta_adj = SOLAR_PANEL_EFF * (1.0 + SOLAR_TEMP_COEFF * (t_cell - SOLAR_NOCT_TEMP))
    eta_adj = np.clip(eta_adj, 0.05, None)
    power   = eta_adj * SOLAR_PANEL_AREA * ghi / 1000.0
    power  += np.random.normal(0, 0.015 * np.maximum(power, 0.1))
    return np.clip(power, 0.0, None)


# ─────────────────────────────────────────────────────────────────────────────
# Wind power — IEC 61400-2 three-region curve
# ─────────────────────────────────────────────────────────────────────────────
def _wind_power(v: np.ndarray) -> np.ndarray:
    p_aero  = BETZ_COEFF * 0.5 * AIR_DENSITY * ROTOR_AREA * v**3 / 1000.0
    p_rated_val = (BETZ_COEFF * 0.5 * AIR_DENSITY * ROTOR_AREA
                   * WIND_RATED**3 / 1000.0)
    p = np.where(v < WIND_CUT_IN,  0.0,
        np.where(v < WIND_RATED,   p_aero,
        np.where(v < WIND_CUT_OUT, p_rated_val, 0.0)))
    p += np.random.normal(0, 0.02 * np.maximum(p, 0.1))
    return np.clip(p, 0.0, None)


# ─────────────────────────────────────────────────────────────────────────────
# Load demand — real-temperature-scaled Indian village profile (IST)
# ─────────────────────────────────────────────────────────────────────────────
def _load_demand(hour: np.ndarray,
                 month: np.ndarray,
                 doy: np.ndarray,
                 temp: np.ndarray,
                 humidity: np.ndarray) -> np.ndarray:
    """
    Uses REAL temperature & humidity (from Open-Meteo) to scale
    cooling loads accurately instead of using monthly averages.

    Profile (IST):
      Base   : 35 kW  (street lights, medical, comm tower, water pump)
      Morning: 80 kW peak at 07h (cooking, pump, school)
      Evening: 140 kW peak at 20h (lighting, fans, TV, mobile)
      Daytime: 30 kW mid (shops, fans)  9–17h
      Agri   : 60 kW pump load Mar–Jun daytime
      Cooling: proportional to Heat Index above 32 C (real temp)
      Diwali / Holi festival spikes
    """
    load = np.full(len(hour), 35.0)

    # Morning peak
    load += 80.0  * np.exp(-0.5 * ((hour - 7.0)  / 1.2) ** 2)
    # Evening peak
    load += 140.0 * np.exp(-0.5 * ((hour - 20.0) / 1.5) ** 2)

    # Daytime mid-load
    daytime = ((hour >= 9) & (hour <= 17)).astype(float)
    load += 30.0 * daytime

    # Agricultural pump (Mar–Jun, daytime)
    agri = np.isin(month, [3, 4, 5, 6]).astype(float)
    load += 60.0 * agri * daytime

    # Cooling load — based on REAL temperature (Gwalior peaks at 47°C)
    # Evaporative coolers kick in above 32°C; ACs above 38°C
    heat_idx = np.clip(temp - 32.0, 0, None)
    cooler_load = 4.0 * heat_idx                  # ~4 kW per degree above 32
    ac_boost    = np.clip(temp - 38.0, 0, None) * 6.0   # premium above 38°C
    load += cooler_load + ac_boost

    # Humidity amplifies discomfort -> fans run longer
    humid_factor = np.clip((humidity - 60.0) / 40.0, 0, 1)  # 0 at 60%, 1 at 100%
    load += 20.0 * humid_factor

    # Festival loads: Diwali (Oct 20–23), Holi (Mar 7–10)
    diwali = ((doy >= 293) & (doy <= 296)).astype(float)
    holi   = ((doy >=  66) & (doy <=  69)).astype(float)
    load  += 50.0 * (diwali + holi)

    # Night-time dip (22:00–05:00) — only base + fans
    night  = ((hour >= 22) | (hour <= 5)).astype(float)
    load  -= 40.0 * night
    load   = np.maximum(load, LOAD_MIN_KW)

    # ±4% metering noise
    load += np.random.normal(0, 0.04 * load)
    return np.clip(load, LOAD_MIN_KW, None)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("  Gwalior Microgrid — Real Weather Data Fetch")
    print("  Source : Open-Meteo Historical Archive (CC BY 4.0)")
    print(f"  Coords : {LAT} N, {LON} E  (Gwalior, MP)")
    print("  Period : 2023-01-01 to 2024-12-31  (2 years hourly)")
    print("=" * 62)

    # 1. Download weather
    d2023 = _fetch_year(2023)
    d2024 = _fetch_year(2024)

    df = pd.concat([_to_df(d2023), _to_df(d2024)], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Fill minor gaps (rare missing sensor readings)
    df["GHI"]        = pd.to_numeric(df["GHI"],        errors="coerce")
    df["wind_speed"] = pd.to_numeric(df["wind_speed"], errors="coerce")
    df["temperature"]= pd.to_numeric(df["temperature"],errors="coerce")
    df["humidity"]   = pd.to_numeric(df["humidity"],   errors="coerce")
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)
    df["GHI"]        = df["GHI"].clip(lower=0)
    df["wind_speed"] = df["wind_speed"].clip(lower=0)

    print(f"\n[fetch] Real weather rows : {len(df):,}")
    print(f"[fetch] Range             : {df['timestamp'].iloc[0]}  "
          f"to  {df['timestamp'].iloc[-1]}")
    print(f"[fetch] GHI   max/mean    : {df['GHI'].max():.1f} / "
          f"{df['GHI'].mean():.1f} W/m2")
    print(f"[fetch] Temp  max/min     : {df['temperature'].max():.1f} / "
          f"{df['temperature'].min():.1f} degC")
    print(f"[fetch] Wind  max/mean    : {df['wind_speed'].max():.1f} / "
          f"{df['wind_speed'].mean():.1f} m/s")
    print(f"[fetch] Humidity mean     : {df['humidity'].mean():.1f} %")

    # 2. Derive power outputs
    print("\n[fetch] Computing solar power (NOCT thermal model, IEC 61215) ...")
    ghi   = df["GHI"].values
    temp  = df["temperature"].values
    wind  = df["wind_speed"].values
    humid = df["humidity"].values
    hour  = df["timestamp"].dt.hour.values
    month = df["timestamp"].dt.month.values
    doy   = df["timestamp"].dt.day_of_year.values

    df["solar_power"] = np.round(_solar_power(ghi, temp), 4)
    print("[fetch] Computing wind power (IEC 61400-2 piecewise curve) ...")
    df["wind_power"]  = np.round(_wind_power(wind), 4)
    print("[fetch] Computing load demand (real-temp-scaled IST village profile) ...")
    df["load_demand"] = np.round(
        _load_demand(hour.astype(float), month, doy, temp, humid), 3)

    # 3. Save
    cols = ["timestamp", "GHI", "wind_speed", "temperature", "humidity",
            "load_demand", "solar_power", "wind_power"]
    df   = df[cols]

    os.makedirs("data", exist_ok=True)
    out_path = DATA_CSV   # data/gwalior_microgrid.csv
    df.to_csv(out_path, index=False)

    print(f"\n[fetch] SAVED {len(df):,} rows  ->  {out_path}")
    print(f"[fetch] Solar max/mean : {df['solar_power'].max():.2f} / "
          f"{df['solar_power'].mean():.2f} kWh")
    print(f"[fetch] Wind  max/mean : {df['wind_power'].max():.2f} / "
          f"{df['wind_power'].mean():.2f} kWh")
    print(f"[fetch] Load  max/mean : {df['load_demand'].max():.1f} / "
          f"{df['load_demand'].mean():.1f} kW")
    print("\nSample rows (first 5):")
    print(df.head().to_string(index=False))
    print("\n[fetch] DONE. Now run:  python train.py")


if __name__ == "__main__":
    main()
