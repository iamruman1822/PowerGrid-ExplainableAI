"""
generate_data.py
----------------
Generates a physics-accurate 2-year hourly dataset calibrated to:

  Location   : Gwalior, Lakhnotikhurd, Madhya Pradesh, India
  Coordinates: Lat 26.193 degN, Lon 78.1487 degE
  Standards  : MNRE village microgrid | CEA | BIS IS 14286 | IEC 61400-2

Climate profile (from IMD normals for Gwalior):
  - Temperature: 33 degC annual mean, peaks at 47 degC in May
  - Solar: ~5.5 kWh/m2/day GHI annual average (excellent solar zone)
  - Wind: avg 3–6 m/s, peaks in SW monsoon Jun–Sep
  - Humidity: 20–40% dry season, 70–90% monsoon

Microgrid specs (MNRE Tier-2, ~200 households + agri):
  - Solar: 25 kWp community array (138.9 m2 @ 18% efficiency)
  - Wind : 10 kW turbine, 7 m rotor dia. (Suzlon S-20 class)
  - Load : 50–400 kW range with IST-based Indian demand profile

Run:
    python generate_data.py
Output:
    data/gwalior_microgrid.csv
"""

import os
import numpy as np
import pandas as pd

# -- constants from config -----------------------------------------------------
from config import (
    DATA_CSV, SEED,
    LATITUDE, LONGITUDE,
    SOLAR_PANEL_AREA, SOLAR_PANEL_EFF, SOLAR_TEMP_COEFF, SOLAR_NOCT_TEMP,
    BETZ_COEFF, AIR_DENSITY, ROTOR_AREA,
    WIND_CUT_IN, WIND_RATED, WIND_CUT_OUT,
    LOAD_MIN_KW,
)

np.random.seed(SEED)
LAT_RAD = np.radians(LATITUDE)


# -----------------------------------------------------------------------------
# 1.  Clear-sky GHI  (ASHRAE simplified solar geometry)
# -----------------------------------------------------------------------------
def _clear_sky_ghi(hour_utc_frac: np.ndarray,
                   day_of_year  : np.ndarray) -> np.ndarray:
    """
    Returns clear-sky GHI (W/m2) using the ASHRAE simplified model.
    hour_utc_frac: fractional UTC hour (IST - 5.5)
    """
    # Solar declination  (Spencer 1971)
    B         = np.radians((360.0 / 365.0) * (day_of_year - 81.0))
    decl      = np.radians(23.45 * np.sin(B))

    # Equation of time (minutes)
    eot       = (9.87 * np.sin(2 * B) - 7.53 * np.cos(B)
                 - 1.5 * np.sin(B)) / 60.0
    # Solar noon offset from UTC for lon=78.1487 E
    lon_offset = LONGITUDE / 15.0            # hours
    solar_time = hour_utc_frac + lon_offset + eot     # fractional hour

    # Hour angle: 0 at solar noon, +/-π at sunrise/sunset
    omega = np.radians(15.0 * (solar_time - 12.0))

    cos_zenith = (np.sin(LAT_RAD) * np.sin(decl) +
                  np.cos(LAT_RAD) * np.cos(decl) * np.cos(omega))
    cos_zenith = np.maximum(cos_zenith, 0.0)

    # Extra-terrestrial irradiance (W/m2)
    G0 = 1367.0 * (1.0 + 0.033 * np.cos(np.radians(360.0 * day_of_year / 365.0)))
    # ASHRAE clearness index τ_b ≃ 0.65 (Gwalior: arid, very clear sky)
    tau_b  = 0.65
    tau_d  = 0.10
    ghi_cs = G0 * cos_zenith * tau_b + G0 * tau_d
    return np.maximum(ghi_cs * cos_zenith, 0.0)


# -----------------------------------------------------------------------------
# 2.  Realistic GHI with clouds & seasons (Gwalior IMD statistics)
# -----------------------------------------------------------------------------
def _realistic_ghi(cs_ghi: np.ndarray, month: np.ndarray) -> np.ndarray:
    """
    Apply monthly cloud-attenuation factors derived from IMD sunshine-hour
    statistics for Gwalior station (WMO 42340).

    Months:        J    F    M    A    M    J    J    A    S    O    N    D
    Sunshine hrs: 8.5  9.1  9.3  9.6  9.8  7.2  4.5  4.2  6.5  8.9  8.8  8.2
    Cloud factor:  cf = sunshine_hrs / potential_daylight_hrs (normalised)
    """
    cf_by_month = {
        1: 0.88,  2: 0.90,  3: 0.89,  4: 0.91,  5: 0.90,  6: 0.68,
        7: 0.42,  8: 0.39,  9: 0.61, 10: 0.87, 11: 0.89, 12: 0.86,
    }
    cf = np.array([cf_by_month[m] for m in month], dtype=float)
    # Add sub-hour stochastic variability
    noise = np.random.beta(5, 1, len(cs_ghi))    # skewed toward 1 (clear)
    actual_cf = np.clip(cf * noise, 0.2, 1.0)
    ghi = cs_ghi * actual_cf
    return np.maximum(ghi, 0.0)


# -----------------------------------------------------------------------------
# 3.  Temperature (IMD Gwalior normals)
# -----------------------------------------------------------------------------
def _temperature(hour_ist: np.ndarray, day_of_year: np.ndarray) -> np.ndarray:
    """
    Monthly mean temperatures for Gwalior (IMD 1991-2020 normals).
    Daily range is +/-7 degC with afternoon peak at ~14–15 h IST.
    """
    monthly_mean = np.array(
        [17.5, 20.3, 26.5, 33.2, 37.8, 35.2,
         30.1, 28.9, 29.8, 27.5, 21.6, 17.2])   #  degC, Jan–Dec
    month = (day_of_year - 1) // 28              # approx month 0-11
    month = np.clip(month, 0, 11)
    base  = monthly_mean[month]
    daily_swing = 7.0 * np.sin(np.pi * (hour_ist - 5) / 18.0)
    daily_swing = np.where((hour_ist >= 5) & (hour_ist <= 23),
                           daily_swing, -3.0)
    temp = base + daily_swing + np.random.normal(0, 1.0, len(hour_ist))
    return temp


# -----------------------------------------------------------------------------
# 4.  Relative Humidity (IMD Gwalior normals)
# -----------------------------------------------------------------------------
def _humidity(month: np.ndarray) -> np.ndarray:
    """
    Monthly mean RH (%) for Gwalior. Monsoon months very humid.
    """
    monthly_rh = np.array(
        [55, 48, 38, 26, 24, 46, 78, 82, 70, 52, 50, 56])  # Jan–Dec
    base = monthly_rh[np.clip(month - 1, 0, 11)]
    rh   = base + np.random.normal(0, 5, len(month))
    return np.clip(rh, 10, 100).astype(float)


# -----------------------------------------------------------------------------
# 5.  Wind speed (Weibull, Gwalior seasonal profile)
# -----------------------------------------------------------------------------
def _wind_speed(month: np.ndarray) -> np.ndarray:
    """
    Gwalior/central MP wind: calm in winter, picks up Apr–Jun pre-monsoon,
    peaks Jun–Aug during SW monsoon (5–8 m/s gusts).
    Weibull shape k=2 (typical for continental sites).
    """
    monthly_mean_wind = np.array(
        [2.5, 2.8, 3.0, 3.5, 4.2, 6.0,
         6.5, 5.8, 4.0, 2.8, 2.4, 2.3])   # m/s, Jan–Dec
    mean_v = monthly_mean_wind[np.clip(month - 1, 0, 11)]
    # Weibull scale param: λ = mean / Γ(1 + 1/k) = mean / 0.8862 (k=2)
    lam    = mean_v / 0.8862
    wind   = np.random.weibull(2.0, len(month)) * lam
    return np.clip(wind, 0.0, WIND_CUT_OUT)


# -----------------------------------------------------------------------------
# 6.  Solar power — IEC 61215 / NOCT thermal model
# -----------------------------------------------------------------------------
def _solar_power(ghi: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    """
    P_pv = eta(T) x A x GHI / 1000  (kWh/h)
    T_cell = T_amb + (NOCT - 20) / 800 x GHI   (NOCT = 45 degC typical)
    eta(T)   = eta_ref x [1 + alpha x (T_cell - T_ref)]
    """
    NOCT    = 45.0   #  degC  (Nominal Operating Cell Temperature)
    t_cell  = temperature + (NOCT - 20.0) / 800.0 * ghi
    eff_adj = SOLAR_PANEL_EFF * (1 + SOLAR_TEMP_COEFF * (t_cell - SOLAR_NOCT_TEMP))
    solar   = np.maximum(eff_adj, 0.05) * SOLAR_PANEL_AREA * ghi / 1000.0
    solar  += np.random.normal(0, 0.02 * np.maximum(solar, 0.1))
    return np.clip(solar, 0.0, None)


# -----------------------------------------------------------------------------
# 7.  Wind power — piecewise turbine curve (IEC 61400-2)
# -----------------------------------------------------------------------------
def _wind_power(wind: np.ndarray) -> np.ndarray:
    """
    Three-region turbine power curve:
      Region 1 (v < cut_in)  : P = 0
      Region 2 (cut_in <= v < rated): P = Betz x ½rhoAv3  (kWh)
      Region 3 (rated <= v < cut_out): P = rated power (constant)
      Shutdown (v >= cut_out) : P = 0
    """
    p_aero  = BETZ_COEFF * 0.5 * AIR_DENSITY * ROTOR_AREA * wind**3 / 1000.0
    p_rated = BETZ_COEFF * 0.5 * AIR_DENSITY * ROTOR_AREA * WIND_RATED**3 / 1000.0

    p = np.where(wind < WIND_CUT_IN,  0.0,
        np.where(wind < WIND_RATED,   p_aero,
        np.where(wind < WIND_CUT_OUT, p_rated, 0.0)))

    p += np.random.normal(0, 0.03 * np.maximum(p, 0.1))
    return np.clip(p, 0.0, None)


# -----------------------------------------------------------------------------
# 8.  Load demand — Indian village profile (IST hours)
# -----------------------------------------------------------------------------
def _load_demand(hour_ist: np.ndarray,
                 month   : np.ndarray,
                 day_of_year: np.ndarray) -> np.ndarray:
    """
    Gwalior village microgrid load (kW) — MNRE Tier-2 electrification.
    ~200 HH x 250 W avg + agri pump load + commercial/street lighting.

    IST load peaks:
      06:00–09:00  Morning (cooking, water pump, school)
      10:00–16:00  Daytime (water pump, fans, street shops) — low-mid
      18:00–22:00  Evening (lighting, fans, TV, mobile charging)
      22:00–05:00  Night   (base only)
    """
    # Base load (always on: medical, comm, street lights)
    load = np.full(len(hour_ist), 35.0)

    # Morning peak (centred 7h IST)
    load += 80.0  * np.exp(-0.5 * ((hour_ist - 7.0)  / 1.2) ** 2)

    # Evening peak (centred 20h IST) — largest in Indian villages
    load += 140.0 * np.exp(-0.5 * ((hour_ist - 20.0) / 1.5) ** 2)

    # Daytime mid-load (fans, shops, 11–17h)
    daytime   = ((hour_ist >= 11) & (hour_ist <= 17)).astype(float)
    load     += 30.0 * daytime

    # Agricultural pump load (seasonal: Mar–Jun, daytime only)
    agri_mo   = np.isin(month, [3, 4, 5, 6]).astype(float)
    load     += 60.0 * agri_mo * daytime

    # Summer cooling (Apr–Jun: evaporative coolers, ACs add ~25%)
    summer    = np.isin(month, [4, 5, 6]).astype(float)
    load     += 0.25 * 35.0 * summer

    # Festival spikes (Diwali ≈ Oct 23–25, Holi ≈ Mar 8–10)
    diwali    = ((day_of_year >= 296) & (day_of_year <= 298)).astype(float)
    holi      = ((day_of_year >= 67)  & (day_of_year <= 69)).astype(float)
    load     += 40.0 * (diwali + holi)

    # Noise +/-5%
    load += np.random.normal(0, 0.05 * load)
    return np.clip(load, LOAD_MIN_KW, None)


# -----------------------------------------------------------------------------
# 9.  Main generator
# -----------------------------------------------------------------------------
def generate_dataset(n_years: int = 2) -> pd.DataFrame:
    n_hours     = n_years * 8760
    start_ist   = pd.Timestamp("2023-01-01 00:00:00", tz="Asia/Kolkata")
    timestamps  = pd.date_range(start_ist, periods=n_hours, freq="h")

    hour_ist    = timestamps.hour.to_numpy()
    # UTC hour (fractional) for solar geometry calculations
    hour_utc    = (hour_ist - 5.5) % 24
    day_of_year = timestamps.day_of_year.to_numpy()
    month       = timestamps.month.to_numpy()

    print("[generate] Computing clear-sky GHI (solar geometry for Gwalior) ...")
    ghi_cs  = _clear_sky_ghi(hour_utc, day_of_year)
    ghi     = _realistic_ghi(ghi_cs, month)

    print("[generate] Computing temperature (IMD Gwalior normals) ...")
    temp    = _temperature(hour_ist.astype(float), day_of_year)

    print("[generate] Computing humidity & wind speed ...")
    rh      = _humidity(month)
    wind    = _wind_speed(month)

    print("[generate] Computing solar power (NOCT thermal model) ...")
    solar   = _solar_power(ghi, temp)

    print("[generate] Computing wind power (IEC 61400-2 curve) ...")
    wpower  = _wind_power(wind)

    print("[generate] Computing load demand (Indian village IST profile) ...")
    load    = _load_demand(hour_ist.astype(float), month, day_of_year)

    # Strip timezone for CSV compatibility
    ts_naive = timestamps.tz_localize(None)

    df = pd.DataFrame({
        "timestamp"   : ts_naive,
        "GHI"         : np.round(ghi,    3),
        "wind_speed"  : np.round(wind,   3),
        "temperature" : np.round(temp,   3),
        "humidity"    : np.round(rh,     3),
        "load_demand" : np.round(load,   3),
        "solar_power" : np.round(solar,  4),
        "wind_power"  : np.round(wpower, 4),
    })

    os.makedirs(os.path.dirname(DATA_CSV), exist_ok=True)
    df.to_csv(DATA_CSV, index=False)

    print(f"\n[generate] DONE — Saved {len(df):,} rows  ->  {DATA_CSV}")
    print(f"[generate] GHI  max/mean : {df['GHI'].max():.1f} / {df['GHI'].mean():.1f} W/m2")
    print(f"[generate] Temp max/mean : {df['temperature'].max():.1f} / {df['temperature'].mean():.1f} degC")
    print(f"[generate] Wind max/mean : {df['wind_speed'].max():.1f} / {df['wind_speed'].mean():.1f} m/s")
    print(f"[generate] Solar max/mean: {df['solar_power'].max():.2f} / {df['solar_power'].mean():.2f} kWh")
    print(f"[generate] Wind  max/mean: {df['wind_power'].max():.2f} / {df['wind_power'].mean():.2f} kWh")
    print(f"[generate] Load  max/mean: {df['load_demand'].max():.1f} / {df['load_demand'].mean():.1f} kW")
    return df


if __name__ == "__main__":
    print("=" * 60)
    print(" Gwalior Microgrid Dataset Generator")
    print(f" Location: Lat {26.193} degN, Lon {78.1487} degE")
    print(f" Standards: MNRE Tier-2 | BIS IS 14286 | IEC 61400-2")
    print("=" * 60)
    df = generate_dataset()
    print("\nSample rows (first 5):")
    print(df.head().to_string(index=False))
    print("\nStatistics:")
    print(df.describe().round(3).to_string())
