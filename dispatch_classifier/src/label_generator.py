"""
Label Generator Module
Applies deterministic physical rules and balances classes with controlled synthetic scenarios.
"""

import numpy as np
import pandas as pd
from .constants import CLASS_NAMES, FEATURE_COLS


def assign_label(row):
    """
    Returns:
        0: CONSERVATIVE
        1: MODERATE
        2: AGGRESSIVE
    """
    soh = row["soh_pct"]
    soc = row["soc_pct"]
    solar = row["solar_power_q50"]
    load = row["load_demand_q50"]
    gen = row["total_gen_q50"]
    temp = row["cell_temp_c"]
    anom = row["is_anomaly_any"]
    diesel = row["diesel_cost_per_kwh"]
    grid = row["grid_available"]
    s_unc = row["solar_unc_score"]
    is_day = row["is_daytime"]
    cover = row["renewable_cover"]

    # 1. CONSERVATIVE (Protect equipment, emergency backup)
    if soh < 73:
        return 0
    if soc < 22:
        return 0
    if anom == 1:
        return 0
    if temp > 50:
        return 0
    if (cover < 0.10) and (diesel > 13.0) and (grid == 0) and (s_unc > 3):
        return 0
    if (soc < 35) and (is_day == 0) and (load > 50):
        return 0

    # 2. AGGRESSIVE (Maximize renewables, run deferrable loads)
    healthy = (soh >= 80) and (temp < 45)
    if (solar > 4.0) and (cover > 0.15) and (soc >= 65) and healthy:
        return 2
    if (solar > 7.0) and (soc >= 55) and healthy:
        return 2
    if (cover > 0.25) and (soc >= 70) and healthy:
        return 2
    if (soc >= 80) and (soh >= 85) and (is_day == 1) and (gen > 3.0):
        return 2

    # 3. MODERATE (Nominal balanced operation)
    return 1


def generate_balanced_dataset(df_feat):
    """
    Assigns labels and augments under-represented classes to ensure balanced learning.
    """
    df = df_feat.copy()
    df["label"] = df.apply(assign_label, axis=1)

    count_per_class = df["label"].value_counts()
    max_count = count_per_class.max()
    min_target = max(int(max_count * 0.35), 1200)

    synth_rows = []
    for cls_idx in range(3):
        current = count_per_class.get(cls_idx, 0)
        needed = max(0, min_target - current)
        if needed == 0:
            continue

        rng = np.random.default_rng(seed=cls_idx + 42)
        n = needed

        if cls_idx == 0:  # CONSERVATIVE
            soh_s = rng.uniform(60, 74, n)
            soc_s = rng.uniform(10, 30, n)
            sol_s = rng.uniform(0.0, 1.0, n)
            win_s = rng.uniform(0.0, 0.3, n)
            load_s = rng.uniform(30, 80, n)
            temp_s = rng.uniform(48, 55, n)
            hrs = rng.integers(20, 24, n)
            months = rng.integers(1, 13, n)
            grid_s = rng.integers(0, 2, n)
            diesel_s = rng.uniform(13.0, 16.0, n)
            anom_s = rng.integers(0, 2, n)

        elif cls_idx == 1:  # MODERATE
            soh_s = rng.uniform(73, 92, n)
            soc_s = rng.uniform(30, 68, n)
            sol_s = rng.uniform(0.5, 4.5, n)
            win_s = rng.uniform(0.1, 0.4, n)
            load_s = rng.uniform(35, 85, n)
            temp_s = rng.uniform(28, 46, n)
            hrs = rng.integers(8, 18, n)
            months = rng.integers(1, 13, n)
            grid_s = rng.integers(0, 2, n)
            diesel_s = rng.uniform(11.0, 14.0, n)
            anom_s = np.zeros(n, dtype=int)

        else:  # AGGRESSIVE
            soh_s = rng.uniform(82, 98, n)
            soc_s = rng.uniform(68, 98, n)
            sol_s = rng.uniform(5.5, 12.0, n)
            win_s = rng.uniform(0.2, 0.8, n)
            load_s = rng.uniform(20, 65, n)
            temp_s = rng.uniform(24, 38, n)
            hrs = rng.integers(10, 15, n)
            months = rng.integers(1, 13, n)
            grid_s = rng.integers(0, 2, n)
            diesel_s = rng.uniform(9.0, 13.0, n)
            anom_s = np.zeros(n, dtype=int)

        gen_s = sol_s + win_s
        eps = 1e-3

        for i in range(n):
            synth_rows.append({
                "solar_power_q10": max(0.0, sol_s[i] * 0.70),
                "solar_power_q50": sol_s[i],
                "solar_power_q90": sol_s[i] * 1.30,
                "wind_power_q10": max(0.0, win_s[i] * 0.60),
                "wind_power_q50": win_s[i],
                "wind_power_q90": win_s[i] * 1.40,
                "load_demand_q10": load_s[i] * 0.85,
                "load_demand_q50": load_s[i],
                "load_demand_q90": load_s[i] * 1.15,
                "total_gen_q50": gen_s[i],
                "net_surplus_q50": gen_s[i] - load_s[i],
                "net_surplus_q10": (sol_s[i] * 0.70 + win_s[i] * 0.60) - load_s[i] * 1.15,
                "surplus_ratio": (gen_s[i] - load_s[i]) / (load_s[i] + eps),
                "renewable_cover": min(gen_s[i] / (load_s[i] + eps), 3.0),
                "solar_uncertainty": sol_s[i] * 0.60,
                "wind_uncertainty": win_s[i] * 0.80,
                "load_uncertainty": load_s[i] * 0.30,
                "solar_unc_score": (sol_s[i] * 0.60) / (sol_s[i] + eps),
                "wind_unc_score": (win_s[i] * 0.80) / (win_s[i] + eps),
                "load_unc_score": 0.30,
                "soh_pct": soh_s[i],
                "soc_pct": soc_s[i],
                "deg_rate": 0.001,
                "cumulative_fade": 0.25,
                "cell_temp_c": temp_s[i],
                "is_anomaly_any": int(anom_s[i]),
                "hour_sin": np.sin(2 * np.pi * hrs[i] / 24),
                "hour_cos": np.cos(2 * np.pi * hrs[i] / 24),
                "month_sin": np.sin(2 * np.pi * months[i] / 12),
                "month_cos": np.cos(2 * np.pi * months[i] / 12),
                "is_daytime": int(6 <= hrs[i] <= 18),
                "is_peak_load": int((7 <= hrs[i] <= 10) or (18 <= hrs[i] <= 22)),
                "diesel_cost_per_kwh": diesel_s[i],
                "grid_available": int(grid_s[i]),
                "label": cls_idx,
            })

    if synth_rows:
        df_augmented = pd.concat([df[FEATURE_COLS + ["label"]], pd.DataFrame(synth_rows)], ignore_index=True)
    else:
        df_augmented = df[FEATURE_COLS + ["label"]].copy()

    print("   -> Class distribution after augmentation:")
    for cls_idx, name in enumerate(CLASS_NAMES):
        count = (df_augmented["label"] == cls_idx).sum()
        pct = count / len(df_augmented) * 100
        print(f"      [{cls_idx}] {name:<12}: {count:,} ({pct:.1f}%)")

    return df_augmented
