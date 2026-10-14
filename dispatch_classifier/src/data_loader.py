"""
Data Loader Module
Loads probabilistic solar/wind/load predictions and NASA battery diagnostics.
"""

import pandas as pd
import numpy as np
from .constants import FORECAST_CSV, BATTERY_CSV


def load_datasets():
    """
    Loads and aligns forecast and battery operational datasets.
    Returns:
        pd.DataFrame with merged feature sets
    """
    print("   -> Loading forecast records from:", FORECAST_CSV)
    df_fc = pd.read_csv(FORECAST_CSV)
    
    print("   -> Loading battery fleet records from:", BATTERY_CSV)
    df_bat = pd.read_csv(BATTERY_CSV)

    # Standardize column naming if needed
    n_samples = min(len(df_fc), len(df_bat))
    df_fc_subset = df_fc.iloc[:n_samples].reset_index(drop=True)
    df_bat_subset = df_bat.iloc[:n_samples].reset_index(drop=True)

    # Merge features side-by-side
    df_merged = pd.concat([df_fc_subset, df_bat_subset], axis=1)
    # Remove duplicate columns if any
    df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]

    print(f"   -> Aligned {len(df_merged)} hourly operational scenarios.")
    return df_merged
