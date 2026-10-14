"""
Microgrid Dispatch Classifier - Standalone Pipeline Entrypoint
=============================================================
Usage:
    python run.py
"""

import sys
import time

from src.data_loader import load_datasets
from src.feature_engineering import engineer_features
from src.label_generator import generate_balanced_dataset
from src.train import train_dispatch_model
from src.explainability import compute_shap_explanations
from src.inference import run_scenario_demo


def main():
    start_time = time.time()
    banner = r"""
======================================================================
   STANDALONE DISPATCH CLASSIFIER & SHAP EXPLAINABILITY ENGINE
   Rural Microgrid Intelligent Energy Management Platform
======================================================================
"""
    print(banner)

    # 1. Load Data
    print("[1/5] Loading public forecast and battery datasets...")
    df_raw = load_datasets()

    # 2. Feature Engineering
    print("\n[2/5] Engineering 34 microgrid operational features...")
    df_feat = engineer_features(df_raw)

    # 3. Label Generation
    print("\n[3/5] Assigning ground-truth physical policies & balancing classes...")
    df_ready = generate_balanced_dataset(df_feat)

    # 4. Train & Evaluate
    print("\n[4/5] Training XGBoost classifier & evaluating performance...")
    model, X_test, y_test = train_dispatch_model(df_ready)

    # 5. SHAP Explainability
    print("\n[5/5] Generating SHAP explainability artifacts...")
    explainer = compute_shap_explanations(model, X_test)

    # 6. Live Scenario Demonstrations
    run_scenario_demo(model, explainer)

    elapsed = time.time() - start_time
    print("\n" + "=" * 68)
    print(f"   EXECUTION COMPLETE in {elapsed:.1f} seconds!")
    print("   Artifacts saved to: ./outputs/")
    print("      - outputs/dispatch_engine.ubj           (Trained model binary)")
    print("      - outputs/confusion_matrix.png          (Evaluation matrix)")
    print("      - outputs/shap_beeswarm_*.png           (3 class beeswarms)")
    print("      - outputs/shap_waterfall_*.png          (3 class waterfalls)")
    print("      - outputs/shap_feature_importance.png   (Global importance)")
    print("      - outputs/shap_force_*.html             (Interactive force plots)")
    print("      - outputs/per_sample_explanations.csv   (Sample driver dataset)")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
