"""
02_train_lstm.py
================
NASA Battery Dataset — Feature Engineering + LSTM Training
Extracts per-cycle features, builds sequences, trains stacked LSTM
to predict capacity (degradation proxy), saves model + scaler.
"""

import os, sys, json, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(DATA_DIR, 'outputs')
PLOT_DIR    = os.path.join(OUTPUT_DIR, 'plots')
DATA_OUT    = os.path.join(OUTPUT_DIR, 'data')
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(DATA_OUT, exist_ok=True)

SEQ_LEN        = 10    # look-back window (cycles)
EPOCHS         = 200
BATCH_SIZE     = 32
PATIENCE       = 25
VAL_SPLIT      = 0.15
TEST_SPLIT     = 0.15
SEED           = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────
# FEATURE COLUMNS (inputs → LSTM)
# ─────────────────────────────────────────────
FEATURE_COLS = [
    'mean_voltage',
    'min_voltage',
    'mean_current',
    'mean_temp',
    'max_temp',
    'discharge_duration',
    'cycle_num_norm',   # normalised cycle index
]
TARGET_COL   = 'capacity'

# ─────────────────────────────────────────────
# LOAD PRE-BUILT CSV (from 01_data_analysis.py)
# ─────────────────────────────────────────────
csv_path = os.path.join(DATA_OUT, 'all_discharge_cycles.csv')
if not os.path.exists(csv_path):
    sys.exit("ERROR: Run 01_data_analysis.py first to generate all_discharge_cycles.csv")

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} discharge cycle records from {len(df['battery'].unique())} batteries.")

# Drop rows with NaN in critical columns
df = df.dropna(subset=FEATURE_COLS[:-1] + [TARGET_COL]).copy()
df = df[df[TARGET_COL] > 0].copy()   # remove zero-capacity outliers

# Normalize cycle_num per battery (0→1 range)
df['cycle_num_norm'] = df.groupby('battery')['cycle_num'].transform(
    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
)
print(f"Records after cleaning: {len(df)}")
print(f"Feature columns: {FEATURE_COLS}")

# ─────────────────────────────────────────────
# SCALE FEATURES + TARGET
# ─────────────────────────────────────────────
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_raw = df[FEATURE_COLS].values
y_raw = df[[TARGET_COL]].values

X_scaled = scaler_X.fit_transform(X_raw)
y_scaled = scaler_y.fit_transform(y_raw)

# Save scalers
pickle.dump(scaler_X, open(os.path.join(DATA_OUT, 'scaler_X.pkl'), 'wb'))
pickle.dump(scaler_y, open(os.path.join(DATA_OUT, 'scaler_y.pkl'), 'wb'))
print("Scalers saved.")

# ─────────────────────────────────────────────
# SEQUENCE BUILDER — per-battery sliding window
# ─────────────────────────────────────────────
def build_sequences(df_sorted, X_scaled_local, y_scaled_local, seq_len):
    """
    For each battery independently, create (seq_len, features) → (1,) pairs.
    Returns X (N, seq_len, features), y (N,), indices, battery labels.
    """
    Xs, ys, idxs, bats = [], [], [], []
    batteries = df_sorted['battery'].unique()
    ptr = 0
    for bat in batteries:
        mask   = df_sorted['battery'].values == bat
        idxs_b = np.where(mask)[0]
        Xb     = X_scaled_local[idxs_b]
        yb     = y_scaled_local[idxs_b]
        for i in range(seq_len, len(Xb)):
            Xs.append(Xb[i - seq_len : i])
            ys.append(yb[i, 0])
            idxs.append(idxs_b[i])
            bats.append(bat)
    return (np.array(Xs, dtype=np.float32),
            np.array(ys, dtype=np.float32),
            np.array(idxs),
            np.array(bats))

df_sorted = df.sort_values(['battery', 'cycle_num']).reset_index(drop=True)

# Re-scale on sorted order
X_sorted  = df_sorted[FEATURE_COLS].values
y_sorted  = df_sorted[[TARGET_COL]].values
X_sc      = scaler_X.transform(X_sorted)
y_sc      = scaler_y.transform(y_sorted)

X_seq, y_seq, seq_idxs, seq_bats = build_sequences(df_sorted, X_sc, y_sc, SEQ_LEN)
print(f"Sequence dataset: X={X_seq.shape}, y={y_seq.shape}")

# ─────────────────────────────────────────────
# TRAIN / VAL / TEST SPLIT  (temporal — no leakage)
# ─────────────────────────────────────────────
n = len(X_seq)
n_test = int(n * TEST_SPLIT)
n_val  = int(n * VAL_SPLIT)
n_train = n - n_val - n_test

X_train = X_seq[:n_train];   y_train = y_seq[:n_train]
X_val   = X_seq[n_train:n_train+n_val]; y_val = y_seq[n_train:n_train+n_val]
X_test  = X_seq[n_train+n_val:];         y_test = y_seq[n_train+n_val:]
test_bats = seq_bats[n_train+n_val:]
test_idxs = seq_idxs[n_train+n_val:]

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ─────────────────────────────────────────────
# LSTM MODEL
# ─────────────────────────────────────────────
def build_model(seq_len, n_features):
    inp = keras.Input(shape=(seq_len, n_features), name='sequence_input')

    x = layers.LSTM(128, return_sequences=True, name='lstm_1')(inp)
    x = layers.Dropout(0.2, name='dropout_1')(x)

    x = layers.LSTM(64, return_sequences=True, name='lstm_2')(x)
    x = layers.Dropout(0.2, name='dropout_2')(x)

    x = layers.LSTM(32, name='lstm_3')(x)
    x = layers.Dropout(0.1, name='dropout_3')(x)

    x = layers.Dense(64, activation='relu', name='dense_1')(x)
    x = layers.Dense(32, activation='relu', name='dense_2')(x)
    out = layers.Dense(1, name='output')(x)

    model = keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    return model

model = build_model(SEQ_LEN, len(FEATURE_COLS))
model.summary()

# ─────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────
cb_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE,
                            restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                patience=10, min_lr=1e-6, verbose=1),
    callbacks.ModelCheckpoint(
        filepath=os.path.join(DATA_DIR, 'lstm_battery_model.keras'),
        monitor='val_loss', save_best_only=True, verbose=0
    ),
]

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
print(f"\nTraining LSTM for up to {EPOCHS} epochs (early stopping patience={PATIENCE}) ...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=cb_list,
    verbose=1,
)

# ─────────────────────────────────────────────
# SAVE TRAINING HISTORY
# ─────────────────────────────────────────────
hist_df = pd.DataFrame(history.history)
hist_df.index.name = 'epoch'
hist_df.to_csv(os.path.join(DATA_OUT, 'training_history.csv'))
print("Training history saved → outputs/data/training_history.csv")

# Save full model (keras format)
model.save(os.path.join(DATA_DIR, 'lstm_battery_model.keras'))
print("Model saved → lstm_battery_model.keras")

# ─────────────────────────────────────────────
# PLOT: TRAINING LOSS CURVES
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(hist_df['loss'],     label='Train MSE', linewidth=2, color='royalblue')
axes[0].plot(hist_df['val_loss'], label='Val MSE',   linewidth=2, color='tomato', linestyle='--')
axes[0].set_title('Training & Validation Loss (MSE)', fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(hist_df['mae'],     label='Train MAE', linewidth=2, color='seagreen')
axes[1].plot(hist_df['val_mae'], label='Val MAE',   linewidth=2, color='darkorange', linestyle='--')
axes[1].set_title('Training & Validation MAE', fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MAE')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.suptitle('LSTM Battery Degradation Model — Training Curves', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '09_training_loss_curves.png'), dpi=150)
plt.close()
print("Plot saved → outputs/plots/09_training_loss_curves.png")

# ─────────────────────────────────────────────
# QUICK TEST EVALUATION
# ─────────────────────────────────────────────
y_pred_sc = model.predict(X_test, verbose=0).flatten()
y_pred    = scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1)).flatten()
y_true    = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

mae   = np.mean(np.abs(y_pred - y_true))
rmse  = np.sqrt(np.mean((y_pred - y_true)**2))
mape  = np.mean(np.abs((y_pred - y_true) / (y_true + 1e-8))) * 100

print(f"\n── Test Set Metrics ──────────────────────────")
print(f"  MAE  : {mae:.4f} Ahr")
print(f"  RMSE : {rmse:.4f} Ahr")
print(f"  MAPE : {mape:.2f} %")

# Save test predictions for use in 03_outputs.py
test_pred_df = pd.DataFrame({
    'battery'  : test_bats,
    'true_cap' : y_true,
    'pred_cap' : y_pred,
    'error'    : y_pred - y_true,
    'abs_error': np.abs(y_pred - y_true),
})
# Also attach cycle info
test_df_rows = df_sorted.iloc[test_idxs].reset_index(drop=True)
test_pred_df['cycle_num'] = test_df_rows['cycle_num'].values
test_pred_df.to_csv(os.path.join(DATA_OUT, 'test_predictions.csv'), index=False)
print("Test predictions saved → outputs/data/test_predictions.csv")

# Save metadata
meta = {
    'seq_len'     : SEQ_LEN,
    'feature_cols': FEATURE_COLS,
    'target_col'  : TARGET_COL,
    'test_mae'    : round(float(mae), 5),
    'test_rmse'   : round(float(rmse), 5),
    'test_mape'   : round(float(mape), 3),
    'epochs_run'  : len(hist_df),
    'best_val_loss': round(float(hist_df['val_loss'].min()), 6),
}
json.dump(meta, open(os.path.join(DATA_OUT, 'model_metadata.json'), 'w'), indent=2)
print("Model metadata saved → outputs/data/model_metadata.json")

print("\n✓ LSTM training complete!")
print(f"  Best val_loss = {meta['best_val_loss']} at epoch {hist_df['val_loss'].idxmin() + 1}")
