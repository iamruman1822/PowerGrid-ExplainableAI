"""
Model Training & Evaluation Module
Trains XGBoost classifier with stratified validation and early stopping.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from .constants import FEATURE_COLS, CLASS_NAMES, OUT_DIR, MODEL_PATH


def train_dispatch_model(df):
    """
    Splits data (80/20 stratified), trains XGBoost, evaluates performance, and saves model.
    """
    X = df[FEATURE_COLS].copy()
    y = df["label"].copy()

    print(f"\n   -> Dataset size: {len(X):,} rows across {len(FEATURE_COLS)} features.")

    # 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Compute class sample weights to guarantee balance
    class_counts = y_train.value_counts()
    total = len(y_train)
    class_weight = {c: total / (len(CLASS_NAMES) * class_counts[c]) for c in class_counts.index}
    sample_weights = y_train.map(class_weight).values

    print("   -> Training XGBoost ensemble (n_estimators=600, max_depth=6, lr=0.05)...")
    model = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.80,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.5,
        reg_lambda=1.5,
        eval_metric="mlogloss",
        early_stopping_rounds=30,
        n_jobs=-1,
        random_state=42,
        num_class=3,
        objective="multi:softprob",
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    print(f"   -> Training finished. Best iteration: {model.best_iteration}")

    # Evaluate
    y_pred = model.predict(X_test).astype(int)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("\n" + "=" * 58)
    print(f"   EVALUATION RESULTS (Held-out Test Set, N={len(y_test):,})")
    print("=" * 58)
    print(f"   Overall Accuracy : {acc * 100:.2f}%")
    print(f"   Weighted F1      : {f1:.4f}")
    print("\n   Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
        linewidths=0.5, linecolor="white"
    )
    ax.set_title("Confusion Matrix - Dispatch Decision Engine", fontsize=12, pad=10)
    ax.set_xlabel("Predicted Strategy", fontsize=10)
    ax.set_ylabel("Ground Truth Strategy", fontsize=10)
    plt.tight_layout()
    cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=140)
    plt.close()
    print(f"   -> Saved: {cm_path}")

    # Save model binary
    model.save_model(MODEL_PATH)
    print(f"   -> Saved model binary: {MODEL_PATH}")

    return model, X_test, y_test
